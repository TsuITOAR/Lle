use super::*;

mod clle;
pub use clle::*;
use iterator_ilp::IteratorILP;
use rustfft::num_traits::zero;

pub trait CoupleOp<T: LleNum> {
    type Linear: LinearOp<T> = NoneOp<T>;
    type NonLinear: NonLinearOp<T> = NoneOp<T>;
    type ConstOp: ConstOp<T> = NoneOp<T>;

    /// input is the freq domain state, start with 0 freq, scaled
    fn linear(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::Linear> {
        None
    }

    fn nonlinear(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::NonLinear> {
        None
    }

    fn constant(&self, _state: &[Complex<T>], _step: Step) -> Option<Self::ConstOp> {
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExtractedCoupleOps<L, N, C> {
    pub linear: Option<L>,
    pub nonlinear: Option<N>,
    pub constant: Option<C>,
}

pub(crate) trait CoupleOpExt<T: LleNum>: CoupleOp<T> {
    fn extract_coup_op(
        &self,
        state: &[Complex<T>],
        step: Step,
    ) -> ExtractedCoupleOps<Self::Linear, Self::NonLinear, Self::ConstOp> {
        #[cfg(feature = "puffin")]
        puffin::profile_function!();
        ExtractedCoupleOps {
            linear: self.linear(state, step),
            nonlinear: self.nonlinear(state, step),
            constant: self.constant(state, step),
        }
    }
}

impl<T: LleNum, C: CoupleOp<T>> CoupleOpExt<T> for C {}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CoupleOpAdd<C1, C2> {
    pub lhs: C1,
    pub rhs: C2,
}

impl<T: LleNum, C1: CoupleOp<T>, C2: CoupleOp<T>> CoupleOp<T> for CoupleOpAdd<C1, C2> {
    type Linear = LinearOpAdd<T, C1::Linear, Option<C2::Linear>>;
    type NonLinear = NonLinearOpAdd<T, C1::NonLinear, Option<C2::NonLinear>>;
    type ConstOp = ConstOpAdd<T, C1::ConstOp, Option<C2::ConstOp>>;
    fn linear(&self, state: &[Complex<T>], step: Step) -> Option<Self::Linear> {
        self.lhs
            .linear(state, step)
            .map(|lhs| lhs.add_linear_op(self.rhs.linear(state, step)))
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.lhs
            .nonlinear(state, step)
            .map(|lhs| lhs.add_nonlin_op(self.rhs.nonlinear(state, step)))
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Self::ConstOp> {
        self.lhs
            .constant(state, step)
            .map(|lhs| lhs.add_const_op(self.rhs.constant(state, step)))
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
    type Linear = C2::Linear;
    type NonLinear = C1::NonLinear;
    type ConstOp = C1::ConstOp;
    fn linear(&self, state: &[Complex<T>], step: Step) -> Option<Self::Linear> {
        self.linear.linear(state, step)
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.couple.nonlinear(state, step)
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Self::ConstOp> {
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
    type Linear = C1::Linear;
    type NonLinear = C2::NonLinear;
    type ConstOp = C1::ConstOp;
    fn linear(&self, state: &[Complex<T>], step: Step) -> Option<Self::Linear> {
        self.couple.linear(state, step)
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.nonlinear.nonlinear(state, step)
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Self::ConstOp> {
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
    type Linear = C1::Linear;
    type NonLinear = C1::NonLinear;
    type ConstOp = C2::ConstOp;
    fn linear(&self, state: &[Complex<T>], step: Step) -> Option<Self::Linear> {
        self.couple.linear(state, step)
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.couple.nonlinear(state, step)
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Self::ConstOp> {
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
    type Linear = C1::Linear;
    type NonLinear = C1::NonLinear;
    type ConstOp = C1::ConstOp;
    fn linear(&self, state: &[Complex<T>], step: Step) -> Option<Self::Linear> {
        self.couple.linear(state, step)
    }
    fn nonlinear(&self, state: &[Complex<T>], step: Step) -> Option<Self::NonLinear> {
        self.couple.nonlinear(state, step)
    }
    fn constant(&self, state: &[Complex<T>], step: Step) -> Option<Self::ConstOp> {
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
    type Linear = Complex<T>;

    type NonLinear = NoneOp<T>;
    fn linear(&self, state: &[Complex<T>], _step: Step) -> Option<Self::Linear> {
        Some(
            Complex::i()
                * T::from_f64(2.).unwrap()
                * state.iter().fold_ilp::<{ crate::ILP_STREAM }, _>(
                    zero,
                    |t: T, x| t + x.norm_sqr(),
                    |t, x| t + x,
                )
                / T::from_usize(state.len()).unwrap(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd, typed_builder::TypedBuilder)]
pub struct ModeSplit<T: LleNum> {
    pub mode: i32,
    pub strength: T,
}

impl<T: LleNum> CoupleOp<T> for ModeSplit<T> {
    fn mix_freq(&self, s1: &mut [Complex<T>], s2: &mut [Complex<T>], step_dist: T) {
        let Self { mode, strength } = self;
        if strength.is_zero() {
            return;
        }
        let mode = mode.rem_euclid(s1.len() as _) as usize;
        let temp = s1[mode];
        s1[mode] += Complex::i() * strength * step_dist * s2[mode].conj();
        s2[mode] += -Complex::i() * strength * step_dist * temp.conj();
        // s1[mode] += Complex::i() * strength * step_dist * s2[mode];
        // s2[mode] += Complex::i() * strength * step_dist * temp;
    }
}
