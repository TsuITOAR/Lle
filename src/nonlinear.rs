use super::*;
use std::marker::PhantomData;

pub trait NonLinearOp<T: LleNum>: Sized {
    fn get_value(&mut self, step: Step, state: &[Complex<T>], dst: &mut [Complex<T>]);
    fn add_nonlin_op<A: NonLinearOp<T>>(self, lhs: A) -> NonLinearOpAdd<T, Self, A> {
        NonLinearOpAdd {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn sub_nonlin_op<A: NonLinearOp<T>>(self, lhs: A) -> NonLinearOpSub<T, Self, A> {
        NonLinearOpSub {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn mul_nonlin_op<A: NonLinearOp<T>>(self, lhs: A) -> NonLinearOpMul<T, Self, A> {
        NonLinearOpMul {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn div_nonlin_op<A: NonLinearOp<T>>(self, lhs: A) -> NonLinearOpDiv<T, Self, A> {
        NonLinearOpDiv {
            op1: self,
            op2: lhs,
            ph: PhantomData,
        }
    }
    fn mul_scalar(self, mul: T) -> NonLinearOpScaled<T, Self> {
        NonLinearOpScaled { op: self, mul }
    }
    fn by_mut(&'_ mut self) -> NonLinearOpMut<'_, Self> {
        NonLinearOpMut { op: self }
    }
    fn skip(&self) -> bool {
        false
    }
}

pub(crate) trait NonLinearOpExt<T: LleNum>: NonLinearOp<T> {
    fn apply(&mut self, state: &mut [Complex<T>], cur_step: Step, step_dist: T) {
        if self.skip() {
            return;
        }
        let len = state.len();
        let mut buf = vec![Complex::zero(); len];

        self.get_value(cur_step, state, &mut buf);
        state
            .iter_mut()
            .zip(buf)
            .for_each(|x| *x.0 *= (x.1 * step_dist).exp())
    }
}

impl<T: LleNum, O: NonLinearOp<T>> NonLinearOpExt<T> for O {}

pub struct NonLinearOpMut<'a, T> {
    op: &'a mut T,
}

impl<T: LleNum, R: NonLinearOp<T>> NonLinearOp<T> for NonLinearOpMut<'_, R> {
    fn get_value(&mut self, step: Step, state: &[Complex<T>], dst: &mut [Complex<T>]) {
        self.op.get_value(step, state, dst);
    }
}

impl<T: LleNum, O: NonLinearOp<T>> NonLinearOp<T> for Option<O> {
    fn get_value(&mut self, step: Step, state: &[Complex<T>], dst: &mut [Complex<T>]) {
        if let Some(op) = self {
            op.get_value(step, state, dst)
        }
    }
    fn skip(&self) -> bool {
        match self {
            Some(x) => x.skip(),
            None => true,
        }
    }
}

pub struct NonLinearOpScaled<T: LleNum, O: NonLinearOp<T>> {
    op: O,
    mul: T,
}
impl<T: LleNum, O: NonLinearOp<T>> NonLinearOp<T> for NonLinearOpScaled<T, O> {
    fn get_value(&mut self, step: Step, state: &[Complex<T>], dst: &mut [Complex<T>]) {
        self.op.get_value(step, state, dst);
        dst.iter_mut().for_each(|x| *x *= self.mul);
    }
}

macro_rules! CompoundNonLinear {
    ($name:ident<$g1:ident, $g2:ident>,$op:tt) => {
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        pub struct $name<T:LleNum,$g1:NonLinearOp<T>,$g2:NonLinearOp<T>> {
            op1:$g1,
            op2:$g2,
            ph:PhantomData<T>
        }
        impl<T:LleNum,$g1:NonLinearOp<T>,$g2:NonLinearOp<T>> NonLinearOp<T> for $name<T,$g1,$g2> {
            #[inline]
            fn get_value(&mut self, step: Step, state: &[Complex<T>], dst: &mut [Complex<T>]){
                if self.op1.skip() || self.op2.skip(){
                    log::info!("NonLinearOp combination includes skippable operation");
                }
                let mut buf1=vec![Complex::<T>::zero();dst.len()];
                let mut buf2=vec![Complex::<T>::zero();dst.len()];
                self.op1.get_value(step,state,&mut buf1);
                self.op2.get_value(step,state,&mut buf2);
                dst.iter_mut().zip(buf1.iter().zip(buf2.iter())).for_each(|(d,(b1,b2))|*d=b1 $op b2);
            }
            fn skip(&self) -> bool{
                self.op1.skip() || self.op2.skip()
            }
        }
    };
}

CompoundNonLinear!( NonLinearOpAdd<P1, P2>,+);
CompoundNonLinear!( NonLinearOpSub<P1, P2>,-);
CompoundNonLinear!( NonLinearOpMul<P1, P2>,*);
CompoundNonLinear!( NonLinearOpDiv<P1, P2>,/);

#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct SPhaMod {}

impl<T: LleNum> NonLinearOp<T> for SPhaMod {
    fn get_value(&mut self, _: Step, state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.iter_mut()
            .zip(state.iter())
            .for_each(|(d, s)| *d = Complex::i() * s.norm_sqr())
    }
}

impl<T: LleNum, F: FnMut(Complex<T>) -> Complex<T>> NonLinearOp<T> for F {
    fn get_value(&mut self, _: Step, state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.iter_mut()
            .zip(state.iter())
            .for_each(|(d, s)| *d = (self)(*s));
    }
}

impl<T: Zero + LleNum> NonLinearOp<T> for NoneOp<T> {
    fn get_value(&mut self, _: Step, _: &[Complex<T>], _: &mut [Complex<T>]) {
        log::info!("NonLinearOp::get_value called on NoneOp")
    }
    fn skip(&self) -> bool {
        true
    }
}
