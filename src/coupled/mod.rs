use super::*;

mod clle;
pub use clle::*;
use rustfft::num_traits::zero;

pub trait CoupleOp<T: LleNum> {
    type LinearR: LinearOp<T> = NoneOp<T>;
    type LinearF: LinearOp<T> = NoneOp<T>;
    type NonLinear: NonLinearOp<T> = NoneOp<T>;

    fn linear_r(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::LinearR> {
        None
    }

    /// input is the freq domain state, start with 0 freq, scaled
    fn linear_f(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::LinearF> {
        None
    }

    fn nonlinear(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::NonLinear> {
        None
    }

    fn constant(&self, _state: &[Complex<T>], _step: Step) -> Option<Complex<T>> {
        None
    }

    fn mix(&self, _s1: &mut [Complex<T>], _s2: &mut [Complex<T>], _step_dist: T) {}

    fn mix_freq(&self, _s1: &mut [Complex<T>], _s2: &mut [Complex<T>], _step_dist: T) {}

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
    fn add<C: CoupleOp<T>>(self, rhs: C) -> CoupleOpAdd<Self, C>
    where
        Self: Sized,
    {
        CoupleOpAdd { lhs: self, rhs }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpAdd<C1, C2> {
    pub lhs: C1,
    pub rhs: C2,
}

impl<T: LleNum, C1: CoupleOp<T>, C2: CoupleOp<T>> CoupleOp<T> for CoupleOpAdd<C1, C2> {
    type LinearR = LinearOpAdd<T, C1::LinearR, Option<C2::LinearR>>;
    type LinearF = LinearOpAdd<T, C1::LinearF, Option<C2::LinearF>>;
    type NonLinear = NonLinearOpAdd<T, C1::NonLinear, Option<C2::NonLinear>>;
    fn linear_r(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearR> {
        self.lhs
            .linear_r(state, step)
            .map(|lhs| lhs.add(self.rhs.linear_r(state, step)))
    }
    fn linear_f(&self, state: &[Complex<T>], step: Step) -> Option<Self::LinearF> {
        self.lhs
            .linear_f(state, step)
            .map(|lhs| lhs.add(self.rhs.linear_f(state, step)))
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.lhs
            .nonlinear(state, step)
            .map(|lhs| lhs.add(self.rhs.nonlinear(state, step)))
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Complex<T>> {
        self.lhs
            .constant(state, step)
            .map(|x| x + self.rhs.constant(state, step).unwrap_or_else(zero))
    }
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.lhs.mix(s1, s2, step_dist);
        self.rhs.mix(s1, s2, step_dist);
    }
    fn mix_freq(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.lhs.mix_freq(s1, s2, step_dist);
        self.rhs.mix_freq(s1, s2, step_dist);
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpWithLinear<C1, C2> {
    pub couple: C1,
    pub linear: C2,
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
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.couple.mix(s1, s2, step_dist)
    }
    fn mix_freq(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.couple.mix_freq(s1, s2, step_dist)
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpWithNonLinear<C1, C2> {
    pub couple: C1,
    pub nonlinear: C2,
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
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.couple.mix(s1, s2, step_dist)
    }
    fn mix_freq(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.couple.mix(s1, s2, step_dist)
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpWithConstant<C1, C2> {
    pub couple: C1,
    pub constant: C2,
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
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.couple.mix(s1, s2, step_dist)
    }
    fn mix_freq(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.couple.mix(s1, s2, step_dist)
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpWithMix<C1, C2> {
    pub couple: C1,
    pub mix: C2,
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
    fn mix(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.mix.mix(s1, s2, step_dist)
    }
    fn mix_freq(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        self.mix.mix_freq(s1, s2, step_dist)
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
    pub mode: i32,
    pub strength: T,
}

impl<T: LleNum> CoupleOp<T> for ModeSplit<T> {
    fn mix_freq(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        let Self { mode, strength } = self;
        let mode = mode.rem_euclid(s1.len() as _) as usize;
        let temp = s1[mode];
        s1[mode] += Complex::i() * strength * step_dist * s2[mode].conj();
        s2[mode] += Complex::i() * strength * step_dist * temp.conj();
        // s1[mode] += Complex::i() * strength * step_dist * s2[mode];
        // s2[mode] += Complex::i() * strength * step_dist * temp;
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
