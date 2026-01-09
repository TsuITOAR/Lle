use iterator_ilp::TrustedLowerBound;
#[cfg(feature = "par")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::*;
use std::marker::PhantomData;

#[cfg(feature = "par")]
mod sync_l {
    pub trait Marker: Sync {}
    impl<T: ?Sized + Sync> Marker for T {}
}

#[cfg(not(feature = "par"))]
mod sync_l {
    pub trait Marker {}
    impl<T: ?Sized> Marker for T {}
}

pub use sync_l::Marker;

/// get_value gives v, and the iterator is give by x*=exp(v*(i dpsi/dtheta)^n dt)=exp(v*freq^n dt)
pub trait LinearOp<T: LleNum>: Sized + Marker {
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T>;

    #[cfg(feature = "par")]
    fn par_get_value_array<const N: usize>(&self, step: Step, freq: &[Freq; N]) -> [Complex<T>; N]
    where
        Self: Sync,
    {
        let mut res = [Complex::zero(); N];
        res.par_iter_mut().zip(freq.par_iter()).for_each(|(x, &f)| {
            *x = self.get_value(step, f);
        });

        res
    }
    #[cfg(feature = "par")]
    fn par_get_value_slice(
        &self,
        step: Step,
        freq: impl IndexedParallelIterator<Item = Freq>,
    ) -> impl IndexedParallelIterator<Item = Complex<T>>
    where
        Self: Sync,
    {
        freq.map(move |f| self.get_value(step, f))
    }

    fn add_linear_op<A: LinearOp<T>>(self, lhs: A) -> LinearOpAdd<T, Self, A> {
        LinearOpAdd {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn sub_linear_op<A: LinearOp<T>>(self, lhs: A) -> LinearOpSub<T, Self, A> {
        LinearOpSub {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn mul_linear_op<A: LinearOp<T>>(self, lhs: A) -> LinearOpMul<T, Self, A> {
        LinearOpMul {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn div_linear_op<A: LinearOp<T>>(self, lhs: A) -> LinearOpDiv<T, Self, A> {
        LinearOpDiv {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn by_ref_linear_op(&'_ self) -> LinearOpRef<'_, Self> {
        LinearOpRef { op: self }
    }
    fn skip(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct LinearOpCached<T> {
    cache: Vec<Complex<T>>,
}

impl<T: LleNum> LinearOpCached<T> {
    pub fn new<L: LinearOp<T> + Sync>(op: L, len: usize) -> Self {
        debug_assert_eq!(
            op.get_value(0, 0),
            op.get_value(1, 0),
            "Only ops independent of step can be cached"
        );
        Self {
            cache: (0..len)
                .into_par_iter()
                .map(|x| op.get_value(0, freq_at(len, x)))
                .collect(),
        }
    }
}

impl<T: LleNum> LinearOp<T> for LinearOpCached<T> {
    #[inline]
    fn get_value(&self, _step: Step, freq: Freq) -> Complex<T> {
        let len = self.cache.len();
        self.cache[freq.rem_euclid(len as _) as usize]
    }
}

pub trait LinearOpExt<T: LleNum>: LinearOp<T> {
    // !WARN:this function will scale every element 'len' times due to fft
    fn apply<S, C>(
        &self,
        state: &mut S,
        const_freq: &C,
        fft: &mut S::FftProcessor,
        step_dist: T,
        cur_step: Step,
    ) where
        S: AsRef<[Complex<T>]> + AsMut<[Complex<T>]> + FftSource<T>,
        C: ConstOp<T>,
    {
        /* if self.skip() && const_freq.skip() {
            return;
        } */

        state.fft_process_forward(fft);
        self.apply_freq(state.as_mut(), step_dist, cur_step);
        const_freq.apply_const_op(state.as_mut(), cur_step, step_dist);
        state.fft_process_inverse(fft);
    }

    // input state_freq should not be fft shifted before
    fn apply_freq(&self, state_freq: &mut [Complex<T>], step_dist: T, cur_step: Step) {
        if self.skip() {
            return;
        }
        shift_freq(state_freq).for_each(|(f, x)| {
            *x *= (self.get_value(cur_step, f) * step_dist).exp();
        });
    }

    #[cfg(feature = "par")]
    fn apply_par<S, C>(
        &self,
        state: &mut S,
        const_freq: &C,
        fft: &mut S::FftProcessor,
        step_dist: T,
        cur_step: Step,
    ) where
        Self: Sync,
        S: AsRef<[Complex<T>]> + AsMut<[Complex<T>]> + FftSource<T> + ?Sized,
        C: ConstOp<T>,
    {
        if self.skip() && const_freq.skip() {
            return;
        }

        state.fft_process_forward(fft);
        self.apply_freq_par(state.as_mut(), step_dist, cur_step);
        const_freq.apply_const_op(state.as_mut(), cur_step, step_dist);
        state.fft_process_inverse(fft);
    }

    #[cfg(feature = "par")]
    // input state_freq should not be fft shifted before
    fn apply_freq_par(&self, state_freq: &mut [Complex<T>], step_dist: T, cur_step: Step)
    where
        Self: Sync,
    {
        if self.skip() {
            return;
        }
        let freq = ((0)..(state_freq.len() as Freq)).into_par_iter();
        let linear = self.par_get_value_slice(cur_step, freq);
        state_freq.par_iter_mut().zip(linear).for_each(|(x, y)| {
            *x *= (y * step_dist).exp();
        });
    }
}

/* use std::iter::{Chain, Enumerate, Map};
use std::slice::IterMut;

pub type ShiftFreqIter<'a, T> = Map<
    Enumerate<Chain<IterMut<'a, T>, IterMut<'a, T>>>,
    impl FnMut((usize, &'a mut T)) -> (Freq, &'a mut T) + 'a,
>; */

pub fn shift_freq<'a, T>(
    freq: &'a mut [T],
) -> impl TrustedLowerBound<Item = (Freq, &'a mut T)> + 'a {
    let len = freq.len();
    let split_pos = len.div_ceil(2); //for odd situations, need to shift (len+1)/2..len, for evens, len/2..len;
    let (pos_freq, neg_freq) = freq.split_at_mut(split_pos);
    neg_freq
        .iter_mut()
        .chain(pos_freq.iter_mut())
        .enumerate()
        .map(move |(i, x)| (i as Freq - (len - split_pos) as Freq, x))
}

impl<T: LleNum, L: LinearOp<T>> LinearOpExt<T> for L {}

#[derive(Debug, Clone)]
pub struct LinearOpRef<'a, T> {
    op: &'a T,
}

impl<T: LleNum, R: LinearOp<T>> LinearOp<T> for LinearOpRef<'_, R> {
    #[inline]
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T> {
        self.op.get_value(step, freq)
    }

    #[cfg(feature = "par")]
    fn par_get_value_array<const N: usize>(&self, step: Step, freq: &[Freq; N]) -> [Complex<T>; N]
    where
        Self: Sync,
    {
        self.op.par_get_value_array(step, freq)
    }
    #[cfg(feature = "par")]
    fn par_get_value_slice(
        &self,
        step: Step,
        freq: impl IndexedParallelIterator<Item = Freq>,
    ) -> impl IndexedParallelIterator<Item = Complex<T>>
    where
        Self: Sync,
    {
        self.op.par_get_value_slice(step, freq)
    }
    fn skip(&self) -> bool {
        self.op.skip()
    }
}

impl<T: LleNum, O: LinearOp<T>> LinearOp<T> for Option<O> {
    #[inline]
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T> {
        self.as_ref()
            .map_or_else(Complex::zero, |x| x.get_value(step, freq))
    }

    fn skip(&self) -> bool {
        match self {
            Some(x) => x.skip(),
            None => true,
        }
    }
}

fn pow_freq<T: LleNum>(freq: Freq, order: DiffOrder) -> Complex<T> {
    Complex::from(T::from_i32(freq.pow(order)).unwrap())
}

impl<T: LleNum> LinearOp<T> for Complex<T> {
    #[inline]
    fn get_value(&self, _: Step, _: Freq) -> Complex<T> {
        *self
    }
}

///
/// - i * (omega_l-omega_l0) = - i * D_n/n! *(l-l0)^n
///
/// this term should be - i * D_n/n!
///
/// (n, - i * D_n / n!)
///
impl<T: LleNum> LinearOp<T> for (DiffOrder, Complex<T>) {
    #[inline]
    fn get_value(&self, _: Step, freq: Freq) -> Complex<T> {
        self.1 * pow_freq(freq, self.0)
    }
}

impl<T: LleNum, F: Fn(Step, Freq) -> Complex<T> + Marker> LinearOp<T> for F {
    #[inline]
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T> {
        self(step, freq)
    }
}

impl<T: LleNum, F: Fn(Step) -> Complex<T> + Marker> LinearOp<T> for (DiffOrder, F) {
    #[inline]
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T> {
        self.1(step) * pow_freq(freq, self.0)
    }
}

pub fn none_op<T>() -> NoneOp<T> {
    NoneOp::default()
}

impl<T> Default for NoneOp<T> {
    fn default() -> Self {
        Self { p: PhantomData }
    }
}

impl<T: Zero + LleNum> LinearOp<T> for NoneOp<T> {
    fn get_value(&self, _step: Step, _freq: Freq) -> Complex<T> {
        log::info!("LinearOp::get_value called on NoneOp");
        Complex::zero()
    }
    fn skip(&self) -> bool {
        true
    }
}

macro_rules! CompoundLinear {
    ($name:ident<$g1:ident, $g2:ident>,$op:tt) => {
        #[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
        pub struct $name<T:LleNum,$g1:LinearOp<T>,$g2:LinearOp<T>> {
            pub op1:$g1,
            pub op2:$g2,
            ph:PhantomData<T>
        }
        impl<T:LleNum,$g1:LinearOp<T>,$g2:LinearOp<T>> LinearOp<T> for $name<T,$g1,$g2> {
            #[inline]
            fn get_value(&self, step: Step, freq: Freq)->Complex<T>{
                self.op1.get_value(step,freq) $op self.op2.get_value(step,freq)
            }
            fn skip(&self)->bool{
                self.op1.skip() && self.op2.skip()
            }
        }
        impl<T:LleNum,$g1:StaticLinearOp<T>,$g2:StaticLinearOp<T>> StaticLinearOp<T> for $name<T,$g1,$g2> {}
    };
}

CompoundLinear!( LinearOpAdd<P1, P2>,+);
CompoundLinear!( LinearOpSub<P1, P2>,-);
CompoundLinear!( LinearOpMul<P1, P2>,*);
CompoundLinear!( LinearOpDiv<P1, P2>,/);

/// Impl for all but SelfPump
pub trait StaticLinearOp<T: LleNum>: LinearOp<T> {
    fn cached_linear_op(self, len: usize) -> LinearOpCached<T>
    where
        Self: Sized + Sync,
    {
        LinearOpCached::new(self, len)
    }
}

impl<T: LleNum> StaticLinearOp<T> for LinearOpCached<T> {
    fn cached_linear_op(self, _len: usize) -> LinearOpCached<T>
    where
        Self: Sized + Sync,
    {
        unreachable!("LinearOpCached should not be used with cached_linear_op")
    }
}

impl<T: LleNum, L: LinearOp<T>> StaticLinearOp<T> for LinearOpRef<'_, L> {}
impl<T: LleNum> StaticLinearOp<T> for NoneOp<T> {}
impl<T: LleNum, L: StaticLinearOp<T>> StaticLinearOp<T> for Option<L> {}
impl<T: LleNum> StaticLinearOp<T> for (DiffOrder, Complex<T>) {}
impl<T: LleNum, F: Fn(Step, Freq) -> Complex<T> + Marker> StaticLinearOp<T> for F {}
impl<T: LleNum, F: Fn(Step) -> Complex<T> + Marker> StaticLinearOp<T> for (DiffOrder, F) {}
impl<T: LleNum> StaticLinearOp<T> for Complex<T> {}
