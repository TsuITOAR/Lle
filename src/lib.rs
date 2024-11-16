#![feature(associated_type_defaults)]
#![feature(type_alias_impl_trait)]
#![feature(portable_simd)]
mod const_term;
mod coupled;
mod linear;
mod lle;
mod nonlinear;

pub use const_term::*;
pub use coupled::*;
pub use linear::*;
pub use lle::*;
pub use nonlinear::*;

use std::sync::Arc;

use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssignRef, NumRef};
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;
use rustfft::{Fft, FftNum};

pub use rustfft::{self, num_complex, num_traits};

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

pub struct BufferedFft<T: LleNum> {
    fft: Arc<dyn Fft<T>>,
    buf: Vec<Complex<T>>,
}

impl<T: LleNum> BufferedFft<T> {
    pub fn new(len: usize) -> (Self, Self) {
        let mut f = FftPlanner::new();
        let fft1 = f.plan_fft_forward(len);
        let fft2 = f.plan_fft_inverse(len);
        (
            Self {
                buf: vec![Complex::zero(); fft1.get_inplace_scratch_len()],
                fft: fft1,
            },
            Self {
                buf: vec![Complex::zero(); fft2.get_inplace_scratch_len()],
                fft: fft2,
            },
        )
    }

    pub fn fft_process(&mut self, data: &mut [Complex<T>]) {
        #[cfg(feature = "puffin")]
        puffin::profile_function!();
        self.fft.process_with_scratch(data, &mut self.buf)
    }
}

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
    constant.apply_linear_op(state, cur_step, step_dist);
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

pub fn apply_linear<T: LleNum, L: LinearOp<T>>(
    state: &mut [Complex<T>],
    linear: &L,
    fft: &mut (BufferedFft<T>, BufferedFft<T>),
    step_dist: T,
    cur_step: Step,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    #[cfg(not(feature = "par"))]
    apply_linear_sync(state, linear, fft, step_dist, cur_step);
    #[cfg(feature = "par")]
    apply_linear_par(state, linear, fft, step_dist, cur_step);
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

pub fn apply_linear_sync<T: LleNum, L: LinearOp<T>>(
    state: &mut [Complex<T>],
    linear: &L,
    fft: &mut (BufferedFft<T>, BufferedFft<T>),
    step_dist: T,
    cur_step: Step,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    linear.apply(state, fft, step_dist, cur_step);
}

#[cfg(feature = "par")]
pub fn apply_linear_par<T: LleNum, L: LinearOp<T> + Sync>(
    state: &mut [Complex<T>],
    linear: &L,
    fft: &mut (BufferedFft<T>, BufferedFft<T>),
    step_dist: T,
    cur_step: Step,
) {
    #[cfg(feature = "puffin")]
    puffin::profile_function!();
    linear.apply_par(state, fft, step_dist, cur_step);
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

pub fn freq_at(len: usize, i: usize) -> Freq {
    let split_pos = (len + 1) / 2;
    ((i + len - split_pos) % len) as Freq - (len - split_pos) as Freq
}

const ILP_STREAM: usize = 8;
