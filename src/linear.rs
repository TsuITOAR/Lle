use super::*;
use std::marker::PhantomData;

/// get_value gives v, and the iterator is give by x*=exp(v*(i dpsi/dtheta)^n dt)=exp(v*freq^n dt)
pub trait LinearOp<T: LleNum>: Sized {
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T>;
    fn add<A: LinearOp<T>>(self, lhs: A) -> LinearOpAdd<T, Self, A> {
        LinearOpAdd {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn sub<A: LinearOp<T>>(self, lhs: A) -> LinearOpSub<T, Self, A> {
        LinearOpSub {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn mul<A: LinearOp<T>>(self, lhs: A) -> LinearOpMul<T, Self, A> {
        LinearOpMul {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn div<A: LinearOp<T>>(self, lhs: A) -> LinearOpDiv<T, Self, A> {
        LinearOpDiv {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn by_ref(&'_ self) -> LinearOpRef<'_, Self> {
        LinearOpRef { op: self }
    }
    fn cached(self, len: usize) -> LinearOpCached<T> {
        LinearOpCached::new(self, len)
    }
    const SKIP: bool = false;
    fn skip(&self) -> bool {
        Self::SKIP
    }
}

#[derive(Debug, Clone)]
pub struct LinearOpCached<T> {
    cache: Vec<Complex<T>>,
}

impl<T: LleNum> LinearOpCached<T> {
    pub fn new<L: LinearOp<T>>(op: L, len: usize) -> Self {
        debug_assert_eq!(
            op.get_value(0, 0),
            op.get_value(1, 0),
            "Only ops independent of step can be cached"
        );
        Self {
            cache: (0..len).map(|x| op.get_value(0, freq_at(len, x))).collect(),
        }
    }
}

impl<T: LleNum> LinearOp<T> for LinearOpCached<T> {
    fn get_value(&self, _step: Step, freq: Freq) -> Complex<T> {
        let len = self.cache.len();
        self.cache[freq.rem_euclid(len as _) as usize]
    }
}

pub(crate) trait LinearOpExt<T: LleNum>: LinearOp<T> {
    // !WARN:this function will scale every element 'len' times due to fft
    fn apply(
        &self,
        state: &mut [Complex<T>],
        fft: &mut (BufferedFft<T>, BufferedFft<T>),
        step_dist: T,
        cur_step: Step,
    ) {
        if self.skip() {
            return;
        }

        fft.0.fft_process(state);
        self.apply_freq(state, step_dist, cur_step);
        fft.1.fft_process(state);
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
}
use std::iter::{Chain, Enumerate, Map};
use std::slice::IterMut;

pub type ShiftFreqIter<'a, T> = Map<
    Enumerate<Chain<std::slice::IterMut<'a, T>, IterMut<'a, T>>>,
    impl FnMut((usize, &'a mut T)) -> (Freq, &'a mut T) + 'a,
>;

pub fn shift_freq<T>(freq: &mut [T]) -> ShiftFreqIter<'_, T> {
    let len = freq.len();
    let split_pos = (len + 1) / 2; //for odd situations, need to shift (len+1)/2..len, for evens, len/2..len;
    let (pos_freq, neg_freq) = freq.split_at_mut(split_pos);
    neg_freq
        .iter_mut()
        .chain(pos_freq.iter_mut())
        .enumerate()
        .map(move |(i, x)| (i as Freq - (len - split_pos) as Freq, x))
}

impl<T: LleNum, L: LinearOp<T>> LinearOpExt<T> for L {}

pub struct LinearOpRef<'a, T> {
    op: &'a T,
}

impl<T: LleNum, R: LinearOp<T>> LinearOp<T> for LinearOpRef<'_, R> {
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T> {
        self.op.get_value(step, freq)
    }
}

impl<T: LleNum, O: LinearOp<T>> LinearOp<T> for Option<O> {
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T> {
        self.as_ref()
            .map_or_else(Complex::zero, |x| x.get_value(step, freq))
    }

    fn skip(&self) -> bool {
        self.is_none()
    }
}

fn pow_freq<T: LleNum>(freq: Freq, order: DiffOrder) -> Complex<T> {
    Complex::from(T::from_i32(freq.pow(order)).unwrap())
}

impl<T: LleNum> LinearOp<T> for Complex<T> {
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
    fn get_value(&self, _: Step, freq: Freq) -> Complex<T> {
        self.1 * pow_freq(freq, self.0)
    }
}

impl<T: LleNum, F: Fn(Step, Freq) -> Complex<T>> LinearOp<T> for F {
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T> {
        self(step, freq)
    }
}

impl<T: LleNum, F: Fn(Step) -> Complex<T>> LinearOp<T> for (DiffOrder, F) {
    fn get_value(&self, step: Step, freq: Freq) -> Complex<T> {
        self.1(step) * pow_freq(freq, self.0)
    }
}

pub fn none_op<T>() -> NoneOp<T> {
    NoneOp::default()
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

impl<T: Zero + LleNum> LinearOp<T> for NoneOp<T> {
    fn get_value(&self, _step: Step, _freq: Freq) -> Complex<T> {
        unreachable!("LinearOp::get_value called on NoneOp")
    }
    const SKIP: bool = true;
}

macro_rules! CompoundLinear {
    ($name:ident<$g1:ident, $g2:ident>,$op:tt) => {
        pub struct $name<T:LleNum,$g1:LinearOp<T>,$g2:LinearOp<T>> {
            op1:$g1,
            op2:$g2,
            ph:PhantomData<T>
        }
        impl<T:LleNum,$g1:LinearOp<T>,$g2:LinearOp<T>> LinearOp<T> for $name<T,$g1,$g2> {
            fn get_value(&self,step:Step,freq:Freq)->Complex<T>{
                self.op1.get_value(step,freq) $op self.op2.get_value(step,freq)
            }
            fn skip(&self)->bool{
                self.op1.skip()||self.op2.skip()
            }
            const SKIP: bool = $g1::SKIP || $g2::SKIP;
        }
    };
}

CompoundLinear!( LinearOpAdd<P1, P2>,+);
CompoundLinear!( LinearOpSub<P1, P2>,-);
CompoundLinear!( LinearOpMul<P1, P2>,*);
CompoundLinear!( LinearOpDiv<P1, P2>,/);

pub trait IntoLinearOps<T: LleNum> {
    type Output: LinearOp<T>;
    fn into_linear_ops(self) -> Option<Self::Output>;
}

impl<T: LleNum, U: LinearOp<T>> IntoLinearOps<T> for U {
    type Output = Self;
    fn into_linear_ops(self) -> Option<Self::Output> {
        Some(self)
    }
}
