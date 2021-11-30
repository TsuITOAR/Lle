#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

use std::num::NonZeroU32;
use std::sync::Arc;

use num::traits::{FloatConst, NumAssignRef};
use num_complex::Complex;
use num_traits::Float;
use rayon::prelude::*;
use rustfft::{num_complex, num_traits, FftPlanner};
use rustfft::{Fft, FftNum};

pub trait LleNum: NumAssignRef + FftNum + Float + FloatConst {}
impl<T> LleNum for T where T: NumAssignRef + FftNum + Float + FloatConst {}

pub trait Evolver<T: LleNum> {
    fn evolve(&mut self);
    fn get_status(&self) -> &[Complex<T>];
    fn evolve_n(&mut self, n: usize) {
        (0..n).for_each(|_| self.evolve())
    }
    fn evolve_until(&mut self, mut until: impl FnMut(&[Complex<T>]) -> bool) {
        while until(self.get_status()) {
            self.evolve();
        }
    }
    fn evolve_with_monitor(&mut self, mut monitor: impl FnMut(&[Complex<T>])) {
        self.evolve();
        monitor(self.get_status())
    }
    fn evolve_n_with_monitor(&mut self, n: usize, mut monitor: impl FnMut(&[Complex<T>])) {
        (0..n).for_each(|_| self.evolve_with_monitor(&mut monitor))
    }
    fn evolve_until_with_monitor(
        &mut self,
        mut until: impl FnMut(&[Complex<T>]) -> bool,
        mut monitor: impl FnMut(&[Complex<T>]),
    ) {
        while until(self.get_status()) {
            self.evolve_with_monitor(&mut monitor)
        }
    }
}

pub trait ParEvolver<T: LleNum> {
    fn par_evolve(&mut self);
    fn get_status(&self) -> &[Complex<T>];
    fn par_evolve_n(&mut self, n: usize) {
        (0..n).for_each(|_| self.par_evolve())
    }
    fn par_evolve_until(&mut self, mut until: impl FnMut(&[Complex<T>]) -> bool) {
        while until(self.get_status()) {
            self.par_evolve();
        }
    }
    fn par_evolve_with_monitor(&mut self, mut monitor: impl FnMut(&[Complex<T>])) {
        self.par_evolve();
        monitor(self.get_status())
    }
    fn par_evolve_n_with_monitor(&mut self, n: usize, mut monitor: impl FnMut(&[Complex<T>])) {
        (0..n).for_each(|_| self.par_evolve_with_monitor(&mut monitor))
    }
    fn par_evolve_until_with_monitor(
        &mut self,
        mut until: impl FnMut(&[Complex<T>]) -> bool,
        mut monitor: impl FnMut(&[Complex<T>]),
    ) {
        while until(self.get_status()) {
            self.par_evolve_with_monitor(&mut monitor)
        }
    }
}

pub struct LleSolver<
    T,
    S,
    Nonlin = Box<dyn Fn(Complex<T>) -> Complex<T>>,
    Linear = [LinearOp<T, NonZeroU32>; 0],
> where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: AsRef<[LinearOp<T, NonZeroU32>]>,
    Nonlin: Fn(Complex<T>) -> Complex<T>,
{
    state: S,
    linear: Option<Linear>,
    nonlin: Option<NonlinOp<Complex<T>, Nonlin>>,
    constant: Option<Complex<T>>,
    step_dist: T,
    fft: (Arc<dyn Fft<T>>, Arc<dyn Fft<T>>),
    len: usize,
    cur_step: u32,
}

impl<T, S, NonLin, Linear> LleSolver<T, S, NonLin, Linear>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    NonLin: Fn(Complex<T>) -> Complex<T>,
    Linear: AsRef<[LinearOp<T, NonZeroU32>]>,
{
    pub fn new<
        L: IntoLinearOps<T, NonZeroU32, Output = Linear>,
        N: Into<Option<NonLin>>,
        C: Into<Option<Complex<T>>>,
    >(
        init: S,
        step_dist: T,
        linear: L,
        nonlin: N,
        constant: C,
    ) -> Self {
        let mut f = FftPlanner::new();
        let len = init.as_ref().len();
        Self {
            state: init,
            linear: linear.to_linear_ops(),
            nonlin: nonlin.into().map(|x| NonlinOp::new(x)),
            constant: constant.into(),
            step_dist,
            fft: (f.plan_fft_forward(len), f.plan_fft_inverse(len)),
            len,
            cur_step: 0,
        }
    }
    pub fn evolve(&mut self) {
        self.cur_step += 1;
        if let Some(ref mut nonlin) = self.nonlin {
            nonlin.refresh(self.state.as_ref());
            self.state
                .as_mut()
                .iter_mut()
                .zip(nonlin.buff().iter())
                .for_each(|x| *x.0 *= (x.1 * self.step_dist).exp())
        }
        if let Some(ref linear) = self.linear {
            apply_linear(
                self.state.as_mut(),
                linear.as_ref(),
                self.len,
                &self.fft,
                self.step_dist,
            );
            let c = self.constant.unwrap_or_else(|| T::zero().into());
            self.state
                .as_mut()
                .iter_mut()
                .for_each(|x| *x = *x / T::from_usize(self.len).unwrap() + c * self.step_dist);
        } else {
            if let Some(c) = self.constant {
                self.state
                    .as_mut()
                    .iter_mut()
                    .for_each(|x| *x += c * self.step_dist)
            }
        }
    }
    pub fn evolve_for(&mut self, steps_num: u32) {
        (0..steps_num).for_each(|_| self.evolve())
    }
    pub fn evolve_for_until(&mut self, mut until: impl FnMut(&[Complex<T>]) -> bool) {
        while until(self.state.as_ref()) {
            self.evolve();
        }
    }
}

impl<T, S, NonLin, Linear> LleSolver<T, S, NonLin, Linear>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    NonLin: Fn(Complex<T>) -> Complex<T> + Sync + Send,
    Linear: AsRef<[LinearOp<T, NonZeroU32>]>,
{
    pub fn par_evolve(&mut self) {
        self.cur_step += 1;
        if let Some(ref mut nonlin) = self.nonlin {
            nonlin.par_refresh(self.state.as_ref());
            self.state
                .as_mut()
                .par_iter_mut()
                .zip(nonlin.buff().par_iter())
                .for_each(|x| *x.0 *= (x.1 * self.step_dist).exp())
        }
        if let Some(ref linear) = self.linear {
            par_apply_linear(
                self.state.as_mut(),
                linear.as_ref(),
                self.len,
                &self.fft,
                self.step_dist,
            );
            let c = self.constant.unwrap_or_else(|| T::zero().into());
            self.state
                .as_mut()
                .par_iter_mut()
                .for_each(|x| *x = *x / T::from_usize(self.len).unwrap() + c * self.step_dist);
        } else {
            if let Some(c) = self.constant {
                self.state
                    .as_mut()
                    .par_iter_mut()
                    .for_each(|x| *x += c * self.step_dist)
            }
        }
    }
    pub fn par_evolve_for(&mut self, steps_num: u32) {
        (0..steps_num).for_each(|_| self.par_evolve())
    }
    pub fn par_evolve_for_until(&mut self, mut until: impl FnMut(&[Complex<T>]) -> bool) {
        while until(self.state.as_ref()) {
            self.par_evolve();
        }
    }
}

// !WARN:this function will scale every element self.len times due to fft
fn apply_linear<T: LleNum>(
    state: &mut [Complex<T>],
    linear: &[LinearOp<T, NonZeroU32>],
    len: usize,
    fft: &(Arc<dyn Fft<T>>, Arc<dyn Fft<T>>),
    step_dist: T,
) {
    let state = state.as_mut();
    let linear = linear.as_ref();
    let split_pos = len / 2;
    fft.0.process(state);
    let (pos_freq, neg_freq) = state.split_at_mut(split_pos);
    let linear = linear.iter();
    neg_freq
        .iter_mut()
        .chain(pos_freq.iter_mut())
        .enumerate()
        .for_each(|x| {
            *x.1 *= (linear.clone().map(|l| {
                l.coefficient * T::from_usize((x.0 - split_pos / 2).pow(l.order.get())).unwrap()
            }))
            .reduce(std::ops::Add::<Complex<T>>::add)
            .map_or_else(
                || Complex::new(T::one(), T::zero()),
                |l| (l * step_dist).exp(),
            )
        });
    fft.1.process(state);
}

// !WARN:this function will scale every element self.len times due to fft
fn par_apply_linear<T: LleNum>(
    state: &mut [Complex<T>],
    linear: &[LinearOp<T, NonZeroU32>],
    len: usize,
    fft: &(Arc<dyn Fft<T>>, Arc<dyn Fft<T>>),
    step_dist: T,
) {
    let state = state.as_mut();
    let linear = linear.as_ref();
    let split_pos = len / 2;
    fft.0.process(state);
    let (pos_freq, neg_freq) = state.split_at_mut(split_pos);
    let linear = linear.iter(); //this should be small enough, so no need to parallel
    neg_freq
        .par_iter_mut()
        .chain(pos_freq.par_iter_mut())
        .enumerate()
        .for_each(|x| {
            *x.1 *= (linear.clone().map(|l| {
                l.coefficient * T::from_usize((x.0 - split_pos / 2).pow(l.order.get())).unwrap()
            }))
            .reduce(std::ops::Add::<Complex<T>>::add)
            .map_or_else(
                || Complex::new(T::one(), T::zero()),
                |l| (l * step_dist).exp(),
            )
        });
    fft.1.process(state);
}

struct NonlinOp<C: Copy, Op: Fn(C) -> C> {
    buff: Vec<C>,
    operator: Op,
}

impl<C: Copy, Op: Fn(C) -> C> NonlinOp<C, Op> {
    fn new(operator: Op) -> Self {
        Self {
            buff: Vec::new(),
            operator,
        }
    }
    fn refresh(&mut self, state: &[C]) {
        self.buff.clear();
        self.buff.extend(state.iter().map(|x| (self.operator)(*x)))
    }
    fn par_refresh(&mut self, state: &[C])
    where
        C: Sync + Send,
        Op: Sync + Send,
    {
        state
            .par_iter()
            .map(|x| (self.operator)(*x))
            .collect_into_vec(&mut self.buff)
    }
    fn buff(&self) -> &[C] {
        &self.buff
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LinearOp<T: LleNum, O> {
    order: O,
    coefficient: Complex<T>,
}

impl<T: LleNum> LinearOp<T, NonZeroU32> {
    fn new(order: NonZeroU32, coefficient: Complex<T>) -> Self {
        Self { order, coefficient }
    }
}

pub trait IntoLinearOps<T: LleNum, O> {
    type Output: AsRef<[LinearOp<T, O>]>;
    fn to_linear_ops(self) -> Option<Self::Output>;
}

impl<T: LleNum, const LEN: usize> IntoLinearOps<T, NonZeroU32> for [(u32, T); LEN] {
    type Output = [LinearOp<T, NonZeroU32>; LEN];
    fn to_linear_ops(self) -> Option<Self::Output> {
        use std::mem::MaybeUninit;
        let mut ret: [MaybeUninit<LinearOp<T, NonZeroU32>>; LEN] =
            unsafe { MaybeUninit::uninit().assume_init() };

        self.into_iter().zip(ret.iter_mut()).for_each(|(x, r)| {
            r.write(LinearOp::new(
                NonZeroU32::new(x.0).expect("linear ops order can't be zero"),
                (x.1).into(),
            ));
        });
        Some(unsafe { ret.as_ptr().cast::<Self::Output>().read() })
    }
}

impl<T: LleNum> IntoLinearOps<T, NonZeroU32> for (u32, T) {
    type Output = [LinearOp<T, NonZeroU32>; 1];
    fn to_linear_ops(self) -> Option<Self::Output> {
        Some([LinearOp::new(
            NonZeroU32::new(self.0).expect("linear ops order can't be zero"),
            (self.1).into(),
        )])
    }
}

impl<T: LleNum> IntoLinearOps<T, NonZeroU32> for Vec<(u32, T)> {
    type Output = Vec<LinearOp<T, NonZeroU32>>;
    fn to_linear_ops(self) -> Option<Self::Output> {
        Some(
            self.into_iter()
                .map(|x| {
                    LinearOp::new(
                        NonZeroU32::new(x.0).expect("linear ops order can't be zero"),
                        (x.1).into(),
                    )
                })
                .collect(),
        )
    }
}

impl<T: LleNum> IntoLinearOps<T, NonZeroU32> for &Vec<(u32, T)> {
    type Output = Vec<LinearOp<T, NonZeroU32>>;
    fn to_linear_ops(self) -> Option<Self::Output> {
        Some(
            self.into_iter()
                .map(|x| {
                    LinearOp::new(
                        NonZeroU32::new(x.0).expect("linear ops order can't be zero"),
                        (x.1).into(),
                    )
                })
                .collect(),
        )
    }
}

impl<T: LleNum> IntoLinearOps<T, NonZeroU32> for &mut Vec<(u32, T)> {
    type Output = Vec<LinearOp<T, NonZeroU32>>;
    fn to_linear_ops(self) -> Option<Self::Output> {
        Some(
            self.into_iter()
                .map(|x| {
                    LinearOp::new(
                        NonZeroU32::new(x.0).expect("linear ops order can't be zero"),
                        (x.1).into(),
                    )
                })
                .collect(),
        )
    }
}

impl<T: LleNum> IntoLinearOps<T, NonZeroU32> for &[(u32, T)] {
    type Output = Vec<LinearOp<T, NonZeroU32>>;
    fn to_linear_ops(self) -> Option<Self::Output> {
        Some(
            self.into_iter()
                .map(|x| {
                    LinearOp::new(
                        NonZeroU32::new(x.0).expect("linear ops order can't be zero"),
                        (x.1).into(),
                    )
                })
                .collect(),
        )
    }
}
impl<T: LleNum> IntoLinearOps<T, NonZeroU32> for &mut [(u32, T)] {
    type Output = Vec<LinearOp<T, NonZeroU32>>;
    fn to_linear_ops(self) -> Option<Self::Output> {
        Some(
            self.into_iter()
                .map(|x| {
                    LinearOp::new(
                        NonZeroU32::new(x.0).expect("linear ops order can't be zero"),
                        (x.1).into(),
                    )
                })
                .collect(),
        )
    }
}

impl<T: LleNum, U: IntoLinearOps<T, NonZeroU32>> IntoLinearOps<T, NonZeroU32> for Option<U> {
    type Output = U::Output;
    fn to_linear_ops(self) -> Option<Self::Output> {
        match self {
            None => None,
            Some(x) => x.to_linear_ops(),
        }
    }
}

/* pub struct CoupledLleSolver<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2, Couple>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: Fn(Complex<T>) -> Complex<T>,
    NonLin1: Fn(Complex<T>) -> Complex<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear2: Fn(Complex<T>) -> Complex<T>,
    NonLin2: Fn(Complex<T>) -> Complex<T>,
    Couple: Fn(&[Complex<T>]) -> Complex<T>,
{
    component1: LleSolver<T, S1, Linear1, NonLin1>,
    component2: LleSolver<T, S2, Linear2, NonLin2>,
    coup_coefficient: Couple,
}
 */
