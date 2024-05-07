use super::*;

mod clle;
pub use clle::*;
use rustfft::num_traits::zero;

pub trait CoupleOp<T: LleNum> {
    type LinearR: LinearOp<T> = NoneOp<T>;
    type LinearF: LinearOp<T> = NoneOp<T>;
    type NonLinear: NonLinearOp<T> = NoneOp<T>;

    /// input is the freq domain state, start with 0 freq, scaled
    fn linear_r(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::LinearR> {
        None
    }
    fn linear_f(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::LinearF> {
        None
    }
    fn nonlinear(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::NonLinear> {
        None
    }
    fn constant(&self, _state: &[Complex<T>], _step: Step) -> Option<Complex<T>> {
        None
    }
    fn mix(&self, _s1: &mut [Complex<T>], _s2: &mut [Complex<T>]) {}
    fn mix_freq(&self, _s1: &mut [Complex<T>], _s2: &mut [Complex<T>]) {}
    fn with_linear<C: CoupleOp<T>>(self, linear: C) -> CoupleOpWithLinear<Self, C>
    where
        Self: Sized,
    {
        CoupleOpWithLinear {
            couple: self,
            linear,
        }
    }
    fn with_nonlinear<C: CoupleOp<T>>(self, nonlinear: C) -> CoupleOpWithNonLinear<Self, C>
    where
        Self: Sized,
    {
        CoupleOpWithNonLinear {
            couple: self,
            nonlinear,
        }
    }
    fn with_constant<C: CoupleOp<T>>(self, constant: C) -> CoupleOpWithConstant<Self, C>
    where
        Self: Sized,
    {
        CoupleOpWithConstant {
            couple: self,
            constant,
        }
    }
    fn with_mix<C: CoupleOp<T>>(self, mix: C) -> CoupleOpWithMix<Self, C>
    where
        Self: Sized,
    {
        CoupleOpWithMix { couple: self, mix }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpWithLinear<C1, C2> {
    couple: C1,
    linear: C2,
}

impl<T: LleNum, C1: CoupleOp<T>, C2: CoupleOp<T>> CoupleOp<T> for CoupleOpWithLinear<C1, C2> {
    type LinearR = C2::LinearR;
    type LinearF = C2::LinearF;
    type NonLinear = C1::NonLinear;
    fn linear_r(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearR> {
        self.linear.linear_r(state, step)
    }
    fn linear_f(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearF> {
        self.linear.linear_f(state, step)
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.couple.nonlinear(state, step)
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Complex<T>> {
        self.couple.constant(state, step)
    }
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>]) {
        self.couple.mix(s1, s2)
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpWithNonLinear<C1, C2> {
    couple: C1,
    nonlinear: C2,
}

impl<T: LleNum, C1: CoupleOp<T>, C2: CoupleOp<T>> CoupleOp<T> for CoupleOpWithNonLinear<C1, C2> {
    type LinearR = C1::LinearR;
    type LinearF = C1::LinearF;
    type NonLinear = C2::NonLinear;
    fn linear_r(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearR> {
        self.couple.linear_r(state, step)
    }
    fn linear_f(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearF> {
        self.couple.linear_f(state, step)
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.nonlinear.nonlinear(state, step)
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Complex<T>> {
        self.couple.constant(state, step)
    }
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>]) {
        self.couple.mix(s1, s2)
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpWithConstant<C1, C2> {
    couple: C1,
    constant: C2,
}

impl<T: LleNum, C1: CoupleOp<T>, C2: CoupleOp<T>> CoupleOp<T> for CoupleOpWithConstant<C1, C2> {
    type LinearR = C1::LinearR;
    type LinearF = C1::LinearF;
    type NonLinear = C1::NonLinear;
    fn linear_r(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearR> {
        self.couple.linear_r(state, step)
    }
    fn linear_f(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearF> {
        self.couple.linear_f(state, step)
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.couple.nonlinear(state, step)
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Complex<T>> {
        self.constant.constant(state, step)
    }
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>]) {
        self.couple.mix(s1, s2)
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpWithMix<C1, C2> {
    couple: C1,
    mix: C2,
}

impl<T: LleNum, C1: CoupleOp<T>, C2: CoupleOp<T>> CoupleOp<T> for CoupleOpWithMix<C1, C2> {
    type LinearR = C1::LinearR;
    type LinearF = C1::LinearF;
    type NonLinear = C1::NonLinear;
    fn linear_r(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearR> {
        self.couple.linear_r(state, step)
    }
    fn linear_f(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearF> {
        self.couple.linear_f(state, step)
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.couple.nonlinear(state, step)
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Complex<T>> {
        self.couple.constant(state, step)
    }
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>]) {
        self.mix.mix(s1, s2)
    }
    fn mix_freq(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>]) {
        self.mix.mix_freq(s1, s2)
    }
}

impl<T: LleNum> CoupleOp<T> for NoneOp<T> {}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct XPhaMod;

impl<T: LleNum> CoupleOp<T> for XPhaMod {
    type LinearR = Complex<T>;

    type NonLinear = NoneOp<T>;
    fn linear_r(&self, state: &[Complex<T>], _step: Step) -> Option<Self::LinearR> {
        Some(
            Complex::i()
                * T::from_f64(2.).unwrap()
                * state.iter().fold(zero(), |t: T, x| t + x.norm_sqr())
                / T::from_usize(state.len()).unwrap(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ModeSplit<T: LleNum> {
    mode: usize,
    strength: T,
}

impl<T: LleNum> CoupleOp<T> for ModeSplit<T> {
    fn constant(&self, state: &[Complex<T>], _step: Step) -> Option<Complex<T>> {
        Some(Complex::i() * self.strength * state[self.mode])
    }
}

#[test]
fn fft_scale_check() {
    let array = [1., 2., 3., 4., 5., 6., 7., 8.];
    let mut fft = rustfft::FftPlanner::new();
    let fft = fft.plan_fft_forward(array.len());
    let mut freq = array
        .iter()
        .map(|x| Complex::new(*x, 0.))
        .collect::<Vec<_>>();
    fft.process(&mut freq);
    let e_real = array.iter().fold(0., |t, x| t + x * x);
    let e_freq = freq.iter().fold(zero(), |t: f64, x| t + x.norm_sqr()) / array.len() as f64;
    assert_eq!(e_real, e_freq);
}
