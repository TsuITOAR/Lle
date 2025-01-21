#![feature(associated_type_defaults)]
#![feature(type_alias_impl_trait)]
#![feature(portable_simd)]
mod const_term;
mod coupled;
mod fft;
mod linear;
mod lle;
mod nonlinear;

pub use const_term::*;
pub use coupled::*;
pub use fft::*;
pub use linear::*;
pub use lle::*;
pub use nonlinear::*;

use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssignRef, NumRef};
use rustfft::num_traits::Zero;

pub use rustfft::{self, num_complex, num_traits, FftNum};

#[cfg(test)]
mod tests;

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
        #[cfg(feature = "puffin")]
        puffin::profile_function!();
        (0..n).for_each(|_| self.evolve())
    }
    fn evolve_until(&mut self, mut until: impl FnMut(&[Complex<T>]) -> bool) {
        #[cfg(feature = "puffin")]
        puffin::profile_function!();
        while !until(self.state()) {
            self.evolve();
        }
    }
    fn evolve_with_monitor(&mut self, mut monitor: impl FnMut(&[Complex<T>])) {
        self.evolve();
        monitor(self.state())
    }
    fn evolve_n_with_monitor(&mut self, n: Step, mut monitor: impl FnMut(&[Complex<T>])) {
        #[cfg(feature = "puffin")]
        puffin::profile_function!();
        (0..n).for_each(|_| self.evolve_with_monitor(&mut monitor))
    }
    fn evolve_until_with_monitor(
        &mut self,
        mut until: impl FnMut(&[Complex<T>]) -> bool,
        mut monitor: impl FnMut(&[Complex<T>]),
    ) {
        #[cfg(feature = "puffin")]
        puffin::profile_function!();
        while !until(self.state()) {
            self.evolve_with_monitor(&mut monitor)
        }
    }
    fn cur_step(&self) -> Step;
}
/*
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
} */

// !WARN:this function will scale every element 'len' times due to fft
/*
fn par_apply_linear<T: LleNum, L: LinearOp<T> + Sync>(
    state: &mut [Complex<T>],
    linear: &L,
    len: usize,
    fft: &mut (BufferedFft<T>, BufferedFft<T>),
    step_dist: T,
    cur_step: Step,
) {
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
 */

pub fn apply_constant<T, C: ConstOp<T>>(
    state: &mut [Complex<T>],
    constant: &'_ C,
    cur_step: Step,
    step_dist: T,
) where
    T: LleNum,
{
    if constant.skip() {
        return;
    }
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    constant.apply_const_op(state, cur_step, step_dist);
}

pub fn apply_constant_scale<T, C: ConstOp<T>>(
    state: &mut [Complex<T>],
    constant: &'_ C,
    scale: T,
    cur_step: Step,
    step_dist: T,
) where
    T: LleNum,
{
    debug_assert!(!scale.is_zero());
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    state.iter_mut().for_each(|x| *x /= scale);
    apply_constant(state, constant, cur_step, step_dist);
}

pub fn apply_linear_and_const_freq<
    T: LleNum,
    S: AsRef<[Complex<T>]> + AsMut<[Complex<T>]> + FftSource<T>,
    L: LinearOp<T>,
    C: ConstOp<T>,
>(
    state: &mut S,
    linear: &L,
    constant_freq: &C,
    fft: &mut S::FftProcessor,
    step_dist: T,
    cur_step: Step,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    #[cfg(not(feature = "par"))]
    apply_linear_and_const_freq_sync(state, linear, constant_freq, fft, step_dist, cur_step);
    #[cfg(feature = "par")]
    apply_linear_and_const_freq_par(state, linear, constant_freq, fft, step_dist, cur_step);
}

pub fn apply_linear_freq<T: LleNum, L: LinearOp<T>>(
    state_freq: &mut [Complex<T>],
    linear: &L,
    step_dist: T,
    cur_step: Step,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    #[cfg(not(feature = "par"))]
    apply_linear_freq_sync(state_freq, linear, step_dist, cur_step);
    #[cfg(feature = "par")]
    apply_linear_freq_par(state_freq, linear, step_dist, cur_step);
}

// input state_freq should not be fft shifted before
pub fn apply_linear_freq_sync<T: LleNum, L: LinearOp<T>>(
    state_freq: &mut [Complex<T>],
    linear: &L,
    step_dist: T,
    cur_step: Step,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();

    linear.apply_freq(state_freq, step_dist, cur_step);
}

#[cfg(feature = "par")]
// input state_freq should not be fft shifted before
pub fn apply_linear_freq_par<T: LleNum, L: LinearOp<T> + Sync>(
    state_freq: &mut [Complex<T>],
    linear: &L,
    step_dist: T,
    cur_step: Step,
) where
    L: Sync,
{
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    linear.apply_freq_par(state_freq, step_dist, cur_step);
}

pub fn apply_linear_and_const_freq_sync<
    T: LleNum,
    S: AsRef<[Complex<T>]> + AsMut<[Complex<T>]> + FftSource<T>,
    L: LinearOp<T>,
    C: ConstOp<T>,
>(
    state: &mut S,
    linear: &L,
    constant_freq: &C,
    fft: &mut S::FftProcessor,
    step_dist: T,
    cur_step: Step,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    linear.apply(state, constant_freq, fft, step_dist, cur_step);
}

#[cfg(feature = "par")]
pub fn apply_linear_and_const_freq_par<
    T: LleNum,
    S: AsRef<[Complex<T>]> + AsMut<[Complex<T>]> + FftSource<T>,
    L: LinearOp<T> + Sync,
>(
    state: &mut S,
    linear: &L,
    constant_freq: &C,
    fft: &mut S::FftProcessor,
    step_dist: T,
    cur_step: Step,
) where
    S::FftProcessor: Sync,
{
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    linear.apply_par(state, constant_freq, fft, step_dist, cur_step);
}

pub fn apply_nonlinear<T: LleNum, NL: NonLinearOp<T>>(
    state: &mut [Complex<T>],
    nonlinear: &mut NL,
    step_dist: T,
    cur_step: Step,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();

    nonlinear.apply(state, cur_step, step_dist);
}

pub fn mix<T: LleNum, C: CoupleOp<T>>(
    c: &mut C,
    state1: &mut [Complex<T>],
    state2: &mut [Complex<T>],
    step_dist: T,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();

    c.mix(state1, state2, step_dist);
}

pub fn mix_freq<T: LleNum, C: CoupleOp<T>>(
    c: &mut C,
    state1: &mut [Complex<T>],
    state2: &mut [Complex<T>],
    step_dist: T,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();

    c.mix_freq(state1, state2, step_dist);
}

pub fn freq_at(len: usize, i: usize) -> Freq {
    let split_pos = (len + 1) / 2;
    ((i + len - split_pos) % len) as Freq - (len - split_pos) as Freq
}

pub fn index_at(len: usize, freq: Freq) -> usize {
    let freq = freq + len as Freq;
    (freq as usize) % len
}

const ILP_STREAM: usize = 8;

#[cfg(test)]
mod test {
    #[test]
    fn index_freq_consistent() {
        let len = 100;
        for i in 0..len {
            assert_eq!(i, super::index_at(len, super::freq_at(len, i)));
        }
        let len = 99;
        for i in 0..len {
            assert_eq!(i, super::index_at(len, super::freq_at(len, i)));
        }
    }
}
