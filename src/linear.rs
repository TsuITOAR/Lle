use super::*;
use std::marker::PhantomData;

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
    const SKIP: bool = false;
}

pub(crate) trait LinearOpExt<T: LleNum>: LinearOp<T> {
    // !WARN:this function will scale every element 'len' times due to fft
    fn apply(
        &self,
        state: &mut [Complex<T>],
        len: usize,
        fft: &mut (BufferedFft<T>, BufferedFft<T>),
        cur_step: Step,
        step_dist: T,
    ) {
        if Self::SKIP {
            return;
        }
        let split_pos = (len + 1) / 2; //for odd situations, need to shift (len+1)/2..len, for evens, len/2..len
        fft.0.process(state);
        let (pos_freq, neg_freq) = state.split_at_mut(split_pos);
        neg_freq
            .iter_mut()
            .chain(pos_freq.iter_mut())
            .enumerate()
            .for_each(|x| {
                *x.1 *= (self.get_value(cur_step, x.0 as i32 - (len - split_pos) as i32)
                    * step_dist)
                    .exp()
            });
        fft.1.process(state);
    }
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
}

fn pow_freq<T: LleNum>(freq: Freq, order: DiffOrder) -> Complex<T> {
    (-Complex::i() * T::from_i32(freq).unwrap()).powu(order)
}

impl<T: LleNum> LinearOp<T> for Complex<T> {
    fn get_value(&self, _: Step, _: Freq) -> Complex<T> {
        *self
    }
}

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
        unreachable!()
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
