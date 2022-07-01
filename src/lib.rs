use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssignRef, NumRef};
use rayon::prelude::*;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;
use rustfft::{Fft, FftNum};

pub use rayon;
pub use rustfft::{self, num_complex, num_traits};

#[cfg(test)]
mod tests {
    use crate::{Evolver, LinearOp, LleSolver, Step};
    use rustfft::num_complex::Complex64;
    #[test]
    fn linear_ops() {
        let linear1 = (1, Complex64::from(1.));
        assert_eq!(linear1.get_value(1, 1), 1. * (-Complex64::i() * 1.).powu(1));
        assert_eq!(linear1.get_value(3, 2), 1. * (-Complex64::i() * 2.).powu(1));

        let linear2 = (3, Complex64::from(2.));
        assert_eq!(linear2.get_value(1, 1), 2. * (-Complex64::i() * 1.).powu(3));
        assert_eq!(linear2.get_value(2, 3), 2. * (-Complex64::i() * 3.).powu(3));

        let linear3 = linear1.clone().add(linear2.clone());
        assert_eq!(
            linear3.get_value(1, 3),
            linear1.get_value(1, 3) + linear2.get_value(1, 3)
        );
        assert_eq!(
            linear3.get_value(5, 7),
            linear1.get_value(5, 7) + linear2.get_value(5, 7)
        );

        let linear4 = (1u32, |step: Step| Complex64::from(step as f64));
        assert_eq!(
            linear4.get_value(2, 4),
            Complex64::from(2.) * (-Complex64::i() * 4.).powu(1)
        );
        assert_eq!(
            linear4.get_value(3, 7),
            Complex64::from(3.) * (-Complex64::i() * 7.).powu(1)
        );

        let linear5 = |step: Step, _pos: i32| Complex64::from(step as f64);
        assert_eq!(linear5.get_value(1, 3), (1.).into());
        assert_eq!(linear5.get_value(6, 3), (6.).into());
    }

    #[test]
    fn create_solver() {
        LleSolver::new([Complex64::new(1., 0.); 128], 0.1, None, None, None);
    }
    #[test]
    fn zero_nonlin_op() {
        let start = [Complex64::new(1., 0.)];
        let mut s = LleSolver::new(start.clone(), 0.1, None, |_| (0.).into(), None);
        s.evolve_n_with_monitor(10, |x| {
            assert_eq!(start, x, "zero nonlin operation should not change state")
        });
    }
    #[test]
    fn zero_linear_op() {
        let start = [Complex64::new(1., 0.); 128];
        let mut s = LleSolver::new(
            start.clone(),
            0.1,
            (1u32, Complex64::from(0.)).add((2, Complex64::from(0.))),
            None,
            None,
        );
        s.evolve_n_with_monitor(10, |x| {
            assert_eq!(start, x, "zero linear operation should not change state")
        });
    }
    #[test]
    fn evolve_some() {
        let step_dist = 0.1;
        let mut start = [Complex64::new(1., 0.); 128];
        start[35] = (0.).into();
        let mut s = LleSolver::new(
            start.clone(),
            step_dist,
            (1, Complex64::from(1.)).add((2, Complex64::from(1.))),
            None,
            None,
        );
        s.evolve();
        assert_ne!(start, s.state(),);
        let nonlin = |x: Complex64| x.sqrt();
        let mut s = LleSolver::new(start.clone(), step_dist, None, nonlin, None);
        s.evolve();
        assert_ne!(start, s.state(),);
        let constant = Complex64::from(1.);
        let mut s = LleSolver::new(start.clone(), step_dist, None, None, constant);
        s.evolve();
        assert_eq!(
            start
                .iter()
                .map(|x| x + step_dist * constant)
                .collect::<Vec<_>>(),
            s.state(),
        );
    }
}

pub type Step = u32;
pub type Freq = i32;
pub type DiffOrder = u32;
pub trait LleNum: NumAssignRef + NumRef + FftNum + Float + FloatConst {}
impl<T> LleNum for T where T: NumAssignRef + NumRef + FftNum + Float + FloatConst {}

pub trait Evolver<T: LleNum> {
    fn evolve(&mut self);
    fn state(&self) -> &[Complex<T>];
    fn state_mut(&mut self) -> &mut [Complex<T>];
    fn evolve_n(&mut self, n: Step) {
        (0..n).for_each(|_| self.evolve())
    }
    fn evolve_until(&mut self, mut until: impl FnMut(&[Complex<T>]) -> bool) {
        while !until(self.state()) {
            self.evolve();
        }
    }
    fn evolve_with_monitor(&mut self, mut monitor: impl FnMut(&[Complex<T>])) {
        self.evolve();
        monitor(self.state())
    }
    fn evolve_n_with_monitor(&mut self, n: Step, mut monitor: impl FnMut(&[Complex<T>])) {
        (0..n).for_each(|_| self.evolve_with_monitor(&mut monitor))
    }
    fn evolve_until_with_monitor(
        &mut self,
        mut until: impl FnMut(&[Complex<T>]) -> bool,
        mut monitor: impl FnMut(&[Complex<T>]),
    ) {
        while !until(self.state()) {
            self.evolve_with_monitor(&mut monitor)
        }
    }
}

impl<T, S, Linear, NonLin> Evolver<T> for LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T = T>,
    NonLin: Fn(Complex<T>) -> Complex<T>,
{
    fn evolve(&mut self) {
        LleSolver::evolve(self)
    }
    fn state(&self) -> &[Complex<T>] {
        LleSolver::state(self)
    }
    fn state_mut(&mut self) -> &mut [Complex<T>] {
        LleSolver::state_mut(self)
    }
}

impl<T, S, Linear, NonLin> ParEvolver<T> for LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T = T> + Sync,
    NonLin: Fn(Complex<T>) -> Complex<T> + Send + Sync,
{
    fn par_evolve(&mut self) {
        LleSolver::par_evolve(self)
    }
    fn state(&self) -> &[Complex<T>] {
        LleSolver::state(self)
    }
    fn state_mut(&mut self) -> &mut [Complex<T>] {
        LleSolver::state_mut(self)
    }
}

pub trait ParEvolver<T: LleNum> {
    fn par_evolve(&mut self);
    fn state(&self) -> &[Complex<T>];
    fn state_mut(&mut self) -> &mut [Complex<T>];
    fn par_evolve_n(&mut self, n: Step) {
        (0..n).for_each(|_| self.par_evolve())
    }
    fn par_evolve_until(&mut self, mut until: impl FnMut(&[Complex<T>]) -> bool) {
        while !until(self.state()) {
            self.par_evolve();
        }
    }
    fn par_evolve_with_monitor(&mut self, mut monitor: impl FnMut(&[Complex<T>])) {
        self.par_evolve();
        monitor(self.state())
    }
    fn par_evolve_n_with_monitor(&mut self, n: Step, mut monitor: impl FnMut(&[Complex<T>])) {
        (0..n).for_each(|_| self.par_evolve_with_monitor(&mut monitor))
    }
    fn par_evolve_until_with_monitor(
        &mut self,
        mut until: impl FnMut(&[Complex<T>]) -> bool,
        mut monitor: impl FnMut(&[Complex<T>]),
    ) {
        while !until(self.state()) {
            self.par_evolve_with_monitor(&mut monitor)
        }
    }
}

pub struct LleSolver<
    T,
    S,
    Linear = (DiffOrder, Complex<T>),
    Nonlin = Box<dyn Fn(Complex<T>) -> Complex<T>>,
> where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T = T>,
    Nonlin: Fn(Complex<T>) -> Complex<T>,
{
    state: S,
    pub linear: Option<Linear>,
    pub nonlin: Option<NonlinOp<T, Nonlin>>,
    pub constant: Option<Complex<T>>,
    pub step_dist: T,
    fft: (Arc<dyn Fft<T>>, Arc<dyn Fft<T>>),
    len: usize,
    cur_step: Step,
}

impl<T, S, Linear, NonLin> LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T = T>,
    NonLin: Fn(Complex<T>) -> Complex<T>,
{
    pub fn new<
        L: IntoLinearOps<T = T, Output = Linear>,
        N: IntoNonlinOps<T, Op = NonLin>,
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
            linear: linear.into_linear_ops(),
            nonlin: nonlin.into_nonlin_ops(),
            constant: constant.into(),
            step_dist,
            fft: (f.plan_fft_forward(len), f.plan_fft_inverse(len)),
            len,
            cur_step: 0,
        }
    }
    fn state(&self) -> &[Complex<T>] {
        self.state.as_ref()
    }
    fn state_mut(&mut self) -> &mut [Complex<T>] {
        self.state.as_mut()
    }
    fn evolve(&mut self) {
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
                linear,
                self.len,
                &self.fft,
                self.step_dist,
                self.cur_step,
            );
            let c = self.constant.unwrap_or_else(|| T::zero().into());
            self.state
                .as_mut()
                .iter_mut()
                .for_each(|x| *x = *x / T::from_usize(self.len).unwrap() + c * self.step_dist);
        } else if let Some(c) = self.constant {
            self.state
                .as_mut()
                .iter_mut()
                .for_each(|x| *x += c * self.step_dist)
        }
        self.cur_step += 1;
    }
}

impl<T, S, Linear, NonLin> LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T = T> + Sync,
    NonLin: Fn(Complex<T>) -> Complex<T> + Sync + Send,
{
    fn par_evolve(&mut self) {
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
                linear,
                self.len,
                &self.fft,
                self.step_dist,
                self.cur_step,
            );
            let c = self.constant.unwrap_or_else(|| T::zero().into());
            self.state
                .as_mut()
                .par_iter_mut()
                .for_each(|x| *x = *x / T::from_usize(self.len).unwrap() + c * self.step_dist);
        } else if let Some(c) = self.constant {
            self.state
                .as_mut()
                .par_iter_mut()
                .for_each(|x| *x += c * self.step_dist)
        }
        self.cur_step += 1;
    }
}

// !WARN:this function will scale every element 'len' times due to fft
fn apply_linear<T: LleNum, L: LinearOp<T = T>>(
    state: &mut [Complex<T>],
    linear: &L,
    len: usize,
    fft: &(Arc<dyn Fft<T>>, Arc<dyn Fft<T>>),
    step_dist: T,
    cur_step: Step,
) {
    let split_pos = (len + 1) / 2; //for odd situations, need to shift (len+1)/2..len, for evens, len/2..len
    fft.0.process(state);
    let (pos_freq, neg_freq) = state.split_at_mut(split_pos);
    neg_freq
        .iter_mut()
        .chain(pos_freq.iter_mut())
        .enumerate()
        .for_each(|x| {
            *x.1 *= (linear.get_value(cur_step, x.0 as i32 - (len - split_pos) as i32) * step_dist)
                .exp()
        });
    fft.1.process(state);
}

// !WARN:this function will scale every element 'len' times due to fft
fn par_apply_linear<T: LleNum, L: LinearOp<T = T> + Sync>(
    state: &mut [Complex<T>],
    linear: &L,
    len: usize,
    fft: &(Arc<dyn Fft<T>>, Arc<dyn Fft<T>>),
    step_dist: T,
    cur_step: Step,
) {
    let state = state.as_mut();
    let split_pos = (len + 1) / 2;
    fft.0.process(state);
    let (pos_freq, neg_freq) = state.split_at_mut(split_pos);
    neg_freq
        .par_iter_mut()
        .chain(pos_freq.par_iter_mut())
        .enumerate()
        .for_each(|x| {
            *x.1 *=
                (linear.get_value(cur_step, x.0 as i32 - (split_pos / 2) as i32) * step_dist).exp()
        });
    fft.1.process(state);
}

#[derive(Clone, Debug)]
pub struct NonlinOp<T: LleNum, Op: Fn(Complex<T>) -> Complex<T>> {
    buff: Vec<Complex<T>>,
    operator: Op,
}

impl<T: LleNum, Op: Fn(Complex<T>) -> Complex<T>> NonlinOp<T, Op> {
    fn new(operator: Op) -> Self {
        Self {
            buff: Vec::new(),
            operator,
        }
    }
    fn refresh(&mut self, state: &[Complex<T>]) {
        self.buff.clear();
        self.buff.extend(state.iter().map(|x| (self.operator)(*x)))
    }
    fn par_refresh(&mut self, state: &[Complex<T>])
    where
        Op: Sync + Send,
    {
        state
            .par_iter()
            .map(|x| (self.operator)(*x))
            .collect_into_vec(&mut self.buff)
    }
    fn buff(&self) -> &[Complex<T>] {
        &self.buff
    }
}

pub trait IntoNonlinOps<T: LleNum>: Sized {
    type Op: Fn(Complex<T>) -> Complex<T>;
    fn into_nonlin_ops(self) -> Option<NonlinOp<T, Self::Op>>;
}

impl<T: LleNum, F: Fn(Complex<T>) -> Complex<T>> IntoNonlinOps<T> for F {
    type Op = Self;
    fn into_nonlin_ops(self) -> Option<NonlinOp<T, Self::Op>> {
        Some(NonlinOp::new(self))
    }
}

impl<T: LleNum> IntoNonlinOps<T> for Option<Box<dyn Fn(Complex<T>) -> Complex<T>>> {
    type Op = Box<dyn Fn(Complex<T>) -> Complex<T>>;
    fn into_nonlin_ops(self) -> Option<NonlinOp<T, Self::Op>> {
        self.map(|p| NonlinOp::new(p))
    }
}

/*
 [LinearOp<T,F>;LEN] can't solve when a array composed of two LinearOps with different F, need to use tuple instead
 maybe define a trait for the behavior of input a time value (the step), and output the operator of linear

TODO: Linear trait finished, Nonlinear operator may depend on current step and pos too, also need a trait instead of a single struct
*/

pub trait LinearOp: Sized {
    type T: LleNum;
    fn get_value(&self, step: Step, freq: Freq) -> Complex<Self::T>;
    fn add<A: LinearOp<T = Self::T>>(self, lhs: A) -> LinearOpAdd<Self, A> {
        LinearOpAdd {
            op1: self,
            op2: lhs,
        }
    }
    fn sub<A: LinearOp<T = Self::T>>(self, lhs: A) -> LinearOpSub<Self, A> {
        LinearOpSub {
            op1: self,
            op2: lhs,
        }
    }
    fn mul<A: LinearOp<T = Self::T>>(self, lhs: A) -> LinearOpMul<Self, A> {
        LinearOpMul {
            op1: self,
            op2: lhs,
        }
    }
    fn div<A: LinearOp<T = Self::T>>(self, lhs: A) -> LinearOpDiv<Self, A> {
        LinearOpDiv {
            op1: self,
            op2: lhs,
        }
    }
}

fn pow_freq<T: LleNum>(freq: Freq, order: DiffOrder) -> Complex<T> {
    (-Complex::i() * T::from_i32(freq).unwrap()).powu(order)
}

impl<T: LleNum> LinearOp for (DiffOrder, Complex<T>) {
    type T = T;
    fn get_value(&self, _: Step, freq: Freq) -> Complex<Self::T> {
        self.1 * pow_freq(freq, self.0)
    }
}

impl<T: LleNum, F: Fn(Step, Freq) -> Complex<T>> LinearOp for F {
    type T = T;
    fn get_value(&self, step: Step, freq: Freq) -> Complex<Self::T> {
        self(step, freq)
    }
}

impl<T: LleNum, F: Fn(Step) -> Complex<T>> LinearOp for (DiffOrder, F) {
    type T = T;
    fn get_value(&self, step: Step, freq: Freq) -> Complex<Self::T> {
        self.1(step) * pow_freq(freq, self.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoneOp<T> {
    p: PhantomData<T>,
}

impl<T> Default for NoneOp<T> {
    fn default() -> Self {
        Self { p: PhantomData }
    }
}

impl<T: Zero + LleNum> LinearOp for NoneOp<T> {
    type T = T;
    fn get_value(&self, _step: Step, _freq: Freq) -> Complex<Self::T> {
        T::zero().into()
    }
}

macro_rules! CompoundLinear {
    ($name:ident<$g1:ident, $g2:ident>,$op:tt) => {
        pub struct $name<$g1:LinearOp,$g2:LinearOp<T=$g1::T>> {
            op1:$g1,
            op2:$g2,
        }
        impl<$g1:LinearOp,$g2:LinearOp<T=$g1::T>> LinearOp for $name<$g1,$g2> {
            type T=$g1::T;
            fn get_value(&self,step:Step,freq:Freq)->Complex<Self::T>{
                self.op1.get_value(step,freq) $op self.op2.get_value(step,freq)
            }
        }
    };
}

CompoundLinear!( LinearOpAdd<P1, P2>,+);
CompoundLinear!( LinearOpSub<P1, P2>,-);
CompoundLinear!( LinearOpMul<P1, P2>,*);
CompoundLinear!( LinearOpDiv<P1, P2>,/);

pub trait IntoLinearOps {
    type T: LleNum;
    type Output: LinearOp<T = Self::T>;
    fn into_linear_ops(self) -> Option<Self::Output>;
}

impl<U: LinearOp> IntoLinearOps for U {
    type T = U::T;
    type Output = Self;
    fn into_linear_ops(self) -> Option<Self::Output> {
        Some(self)
    }
}

impl<T: LleNum> IntoLinearOps for Option<(DiffOrder, Complex<T>)> {
    type T = T;
    type Output = (DiffOrder, Complex<T>);
    fn into_linear_ops(self) -> Option<Self::Output> {
        self
    }
}

pub struct CoupledLleSolver<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: LinearOp<T = T>,
    NonLin1: Fn(Complex<T>) -> Complex<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear2: LinearOp<T = T>,
    NonLin2: Fn(Complex<T>) -> Complex<T>,
{
    pub component1: LleSolver<T, S1, Linear1, NonLin1>,
    pub component2: LleSolver<T, S2, Linear2, NonLin2>,
    pub coup_coefficient: Complex<T>,
    cur_step: usize,
}

impl<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2>
    CoupledLleSolver<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: LinearOp<T = T>,
    NonLin1: Fn(Complex<T>) -> Complex<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear2: LinearOp<T = T>,
    NonLin2: Fn(Complex<T>) -> Complex<T>,
{
    pub fn new(
        comp1: LleSolver<T, S1, Linear1, NonLin1>,
        comp2: LleSolver<T, S2, Linear2, NonLin2>,
        coup: impl Into<Complex<T>>,
    ) -> Self {
        Self {
            component1: comp1,
            component2: comp2,
            coup_coefficient: coup.into(),
            cur_step: 0,
        }
    }
}

impl<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2> Evolver<T>
    for CoupledLleSolver<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: LinearOp<T = T>,
    NonLin1: Fn(Complex<T>) -> Complex<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear2: LinearOp<T = T>,
    NonLin2: Fn(Complex<T>) -> Complex<T>,
{
    fn evolve(&mut self) {
        let Self {
            component1,
            component2,
            coup_coefficient,
            cur_step,
        } = self;
        let mut comp1_sqr_ave = component1
            .state()
            .iter()
            .fold(T::zero(), |a, b| a + b.norm_sqr());
        comp1_sqr_ave /= T::from_usize(component1.state().len()).unwrap();
        let mut comp2_sqr_ave = component2
            .state()
            .iter()
            .fold(T::zero(), |a, b| a + b.norm_sqr());
        comp2_sqr_ave /= T::from_usize(component2.state().len()).unwrap();

        fn evolve_comp<
            T: LleNum,
            S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
            Linear: LinearOp<T = T>,
            NonLin: Fn(Complex<T>) -> Complex<T>,
        >(
            s: &mut LleSolver<T, S, Linear, NonLin>,
            counter_sqr_ave: T,
        ) {
            if let Some(ref mut nonlin) = s.nonlin {
                nonlin.refresh(s.state.as_ref());
                s.state
                    .as_mut()
                    .iter_mut()
                    .zip(nonlin.buff().iter())
                    .for_each(|x| {
                        *x.0 *= ((x.1 + Complex::i() * counter_sqr_ave * T::from_f64(2.).unwrap())
                            * s.step_dist)
                            .exp()
                    })
            } else {
                s.state.as_mut().iter_mut().for_each(|x| {
                    *x *= (Complex::i() * counter_sqr_ave * T::from_f64(2.).unwrap() * s.step_dist)
                        .exp()
                })
            }
            if let Some(ref linear) = s.linear {
                apply_linear(
                    s.state.as_mut(),
                    linear,
                    s.len,
                    &s.fft,
                    s.step_dist,
                    s.cur_step,
                );
                s.state
                    .as_mut()
                    .iter_mut()
                    .for_each(|x| *x = *x / T::from_usize(s.len).unwrap());
            };
            if let Some(c) = s.constant {
                s.state
                    .as_mut()
                    .iter_mut()
                    .for_each(|x| *x += c * s.step_dist);
            }
        }
        evolve_comp(component1, comp2_sqr_ave);
        evolve_comp(component2, comp1_sqr_ave);

        //couple term
        component1.fft.0.process(component1.state.as_mut());
        component2.fft.0.process(component2.state.as_mut());
        let s1: Box<[Complex<T>]> = component1.state.as_ref().into();
        let s2: Box<[Complex<T>]> = component2.state.as_ref().into();
        
        component1
            .state
            .as_mut()
            .iter_mut()
            .zip(s2.into_iter())
            .for_each(|(x, y)| *x += *coup_coefficient * Complex::i() * y * component1.step_dist);
        component2
            .state
            .as_mut()
            .iter_mut()
            .zip(s1.into_iter())
            .for_each(|(x, y)| *x += *coup_coefficient * Complex::i() * y * component2.step_dist);

        component1.fft.1.process(component1.state.as_mut());
        component2.fft.1.process(component2.state.as_mut());

        component1
            .state
            .as_mut()
            .iter_mut()
            .for_each(|x| *x = *x / T::from_usize(component1.len).unwrap());
        component2
            .state
            .as_mut()
            .iter_mut()
            .for_each(|x| *x = *x / T::from_usize(component2.len).unwrap());
            
        *cur_step += 1;
    }

    fn state(&self) -> &[Complex<T>] {
        self.component1.state()
    }

    fn state_mut(&mut self) -> &mut [Complex<T>] {
        self.component1.state_mut()
    }
}
