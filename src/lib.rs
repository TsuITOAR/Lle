mod coupled;
mod linear;
mod lle;
mod nonlinear;

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
mod tests {
    use crate::{Evolver, LinearOp, LleSolver, NoneOp, Step};
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

    /*  #[test]
    fn create_solver() {
        LleSolver::new([Complex64::new(1., 0.); 128], 0.1, none_op(), None, None);
    } */
    #[test]
    fn zero_nonlin_op() {
        let start = [Complex64::new(1., 0.)];
        let mut s = LleSolver::<_, _>::builder()
            .state(start.clone())
            .step_dist(0.1)
            .build();
        s.evolve_n_with_monitor(10, |x| {
            assert_eq!(start, x, "zero nonlin operation should not change state")
        });
    }
    #[test]
    fn zero_linear_op() {
        let start = [Complex64::new(1., 0.); 128];
        let mut s = LleSolver::<_, _, _>::builder()
            .state(start.clone())
            .step_dist(0.1)
            .linear((1u32, Complex64::from(0.)).add((2, Complex64::from(0.))))
            .build();

        s.evolve_n_with_monitor(10, |x| {
            assert_eq!(start, x, "zero linear operation should not change state")
        });
    }
    #[test]
    fn evolve_some() {
        let step_dist = 0.1;
        let mut start = [Complex64::new(1., 0.); 128];
        start[35] = (0.).into();
        let mut s = LleSolver::<_, _, _>::builder()
            .state(start.clone())
            .step_dist(0.1)
            .linear((1u32, Complex64::from(1.)).add((2, Complex64::from(1.))))
            .build();

        s.evolve();
        assert_ne!(start, s.state(),);
        let nonlin = |x: Complex64| x.sqrt();
        let mut s = LleSolver::<_, _, NoneOp<_>, _>::builder()
            .state(start.clone())
            .step_dist(0.1)
            .nonlin(nonlin)
            .build();
        s.evolve();
        assert_ne!(start, s.state(),);
        let constant = Complex64::from(1.);
        let mut s = LleSolver::<_, _>::builder()
            .state(start.clone())
            .step_dist(0.1)
            .constant(constant)
            .build();
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

pub(crate) struct BufferedFft<T: LleNum> {
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

    pub fn process(&mut self, data: &mut [Complex<T>]) {
        self.fft.process_with_scratch(data, &mut self.buf)
    }
}

// !WARN:this function will scale every element 'len' times due to fft
fn apply_linear<T: LleNum, L: LinearOp<T>>(
    state: &mut [Complex<T>],
    linear: &L,
    len: usize,
    fft: &mut (BufferedFft<T>, BufferedFft<T>),
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