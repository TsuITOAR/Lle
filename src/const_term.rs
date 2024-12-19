use crate::{LleNum, NoneOp, Step};
use num_complex::Complex;
use rustfft::num_traits::zero;
use std::marker::PhantomData;

pub trait ConstOp<T: LleNum> {
    fn get_value(&self, cur_step: Step, pos: usize) -> Complex<T>;

    fn add_const_op<R: ConstOp<T>>(self, rhs: R) -> ConstOpAdd<T, Self, R>
    where
        Self: Sized,
    {
        ConstOpAdd {
            op1: self,
            op2: rhs,
            ph: PhantomData,
        }
    }

    fn sub_const_op<R: ConstOp<T>>(self, rhs: R) -> ConstOpSub<T, Self, R>
    where
        Self: Sized,
    {
        ConstOpSub {
            op1: self,
            op2: rhs,
            ph: PhantomData,
        }
    }

    fn mul_const_op<R: ConstOp<T>>(self, rhs: R) -> ConstOpMul<T, Self, R>
    where
        Self: Sized,
    {
        ConstOpMul {
            op1: self,
            op2: rhs,
            ph: PhantomData,
        }
    }

    fn div_const_op<R: ConstOp<T>>(self, rhs: R) -> ConstOpDiv<T, Self, R>
    where
        Self: Sized,
    {
        ConstOpDiv {
            op1: self,
            op2: rhs,
            ph: PhantomData,
        }
    }

    fn by_ref_const_op(&'_ self) -> ConstOpRef<'_, T, Self> {
        ConstOpRef {
            op: self,
            ph: PhantomData,
        }
    }

    fn cached_linear_op(self, len: usize) -> ConstOpCached<T>
    where
        Self: Sized,
    {
        ConstOpCached::new(self, len)
    }

    fn get_value_array(&self, cur_step: Step, len: usize) -> Vec<Complex<T>> {
        (0..len).map(|x| self.get_value(cur_step, x)).collect()
    }

    fn fill_value_array(&self, cur_step: Step, dst: &mut [Complex<T>]) {
        dst.iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.get_value(cur_step, i);
        });
    }

    fn apply_linear_op(&self, state: &mut [Complex<T>], cur_step: Step, step_dist: T) {
        state.iter_mut().enumerate().for_each(|(i, x)| {
            *x += self.get_value(cur_step, i) * step_dist;
        });
    }
    fn skip(&self) -> bool {
        false
    }
}

macro_rules! CompoundConst {
    ($name:ident<$g1:ident, $g2:ident>,$op:tt) => {
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        pub struct $name<T:LleNum,$g1:ConstOp<T>,$g2:ConstOp<T>> {
            op1:$g1,
            op2:$g2,
            ph:PhantomData<T>
        }
        impl<T:LleNum,$g1:ConstOp<T>,$g2:ConstOp<T>> ConstOp<T> for $name<T,$g1,$g2> {
            #[inline]
            fn get_value(&self, step: Step, pos: usize)->Complex<T>{
                self.op1.get_value(step,pos) $op self.op2.get_value(step,pos)
            }
            fn skip(&self)->bool{
                self.op1.skip() && self.op2.skip()
            }
        }
    };
}

CompoundConst!( ConstOpAdd<P1, P2>,+);
CompoundConst!( ConstOpSub<P1, P2>,-);
CompoundConst!( ConstOpMul<P1, P2>,*);
CompoundConst!( ConstOpDiv<P1, P2>,/);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstOpCached<T: LleNum> {
    cache: Vec<Complex<T>>,
}

impl<T: LleNum> ConstOpCached<T> {
    pub fn new<L: ConstOp<T>>(op: L, len: usize) -> Self {
        Self {
            cache: op.get_value_array(Step::default(), len),
        }
    }

    pub fn get_cache(&self) -> &[Complex<T>] {
        &self.cache
    }
}

impl<T: LleNum> ConstOp<T> for ConstOpCached<T> {
    fn get_value(&self, _cur_step: Step, pos: usize) -> Complex<T> {
        self.cache[pos]
    }
    fn fill_value_array(&self, _cur_step: Step, dst: &mut [Complex<T>]) {
        dst.copy_from_slice(&self.cache);
    }
    fn get_value_array(&self, _cur_step: Step, _len: usize) -> Vec<Complex<T>> {
        self.cache.clone()
    }
    fn apply_linear_op(&self, state: &mut [Complex<T>], _cur_step: Step, step_dist: T) {
        state.iter_mut().zip(self.cache.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

pub struct ConstOpRef<'a, T: LleNum, L: ConstOp<T> + ?Sized> {
    op: &'a L,
    ph: PhantomData<T>,
}

impl<T: LleNum, L: ConstOp<T>> ConstOp<T> for ConstOpRef<'_, T, L> {
    fn get_value(&self, cur_step: Step, pos: usize) -> Complex<T> {
        self.op.get_value(cur_step, pos)
    }
    fn skip(&self) -> bool {
        self.op.skip()
    }
}

impl<T: LleNum> ConstOp<T> for Complex<T> {
    fn get_value(&self, _cur_step: Step, _pos: usize) -> Complex<T> {
        *self
    }
}

impl ConstOp<f64> for f64 {
    fn get_value(&self, _cur_step: Step, _pos: usize) -> Complex<f64> {
        Complex::new(*self, 0.)
    }
}

impl ConstOp<f32> for f32 {
    fn get_value(&self, _cur_step: Step, _pos: usize) -> Complex<f32> {
        Complex::new(*self, 0.)
    }
}

impl<T: LleNum> ConstOp<T> for [Complex<T>] {
    fn get_value(&self, _cur_step: Step, pos: usize) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _cur_step: Step, dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self);
    }
    fn get_value_array(&self, _cur_step: Step, _len: usize) -> Vec<Complex<T>> {
        self.to_vec()
    }
    fn apply_linear_op(&self, state: &mut [Complex<T>], _cur_step: Step, step_dist: T) {
        state.iter_mut().zip(self).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum> ConstOp<T> for Vec<Complex<T>> {
    fn get_value(&self, _cur_step: Step, pos: usize) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _cur_step: Step, dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self);
    }
    fn get_value_array(&self, _cur_step: Step, _len: usize) -> Vec<Complex<T>> {
        self.clone()
    }
    fn apply_linear_op(&self, state: &mut [Complex<T>], _cur_step: Step, step_dist: T) {
        state.iter_mut().zip(self).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum> ConstOp<T> for &Vec<Complex<T>> {
    fn get_value(&self, _step: Step, pos: usize) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self);
    }
    fn get_value_array(&self, _step: Step, _len: usize) -> Vec<Complex<T>> {
        (*self).clone()
    }
    fn apply_linear_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum> ConstOp<T> for &[Complex<T>] {
    fn get_value(&self, _step: Step, pos: usize) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self);
    }
    fn get_value_array(&self, _step: Step, _len: usize) -> Vec<Complex<T>> {
        self.to_vec()
    }
    fn apply_linear_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum, const L: usize> ConstOp<T> for [Complex<T>; L] {
    fn get_value(&self, _step: Step, pos: usize) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self.as_slice());
    }
    fn get_value_array(&self, _step: Step, _len: usize) -> Vec<Complex<T>> {
        self.to_vec()
    }
    fn apply_linear_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum, const L: usize> ConstOp<T> for &'_ [Complex<T>; L] {
    fn get_value(&self, _step: Step, pos: usize) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self.as_slice());
    }
    fn get_value_array(&self, _step: Step, _len: usize) -> Vec<Complex<T>> {
        self.to_vec()
    }
    fn apply_linear_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum> ConstOp<T> for NoneOp<T> {
    fn get_value(&self, _step: Step, _pos: usize) -> Complex<T> {
        unreachable!()
    }
    fn skip(&self) -> bool {
        true
    }
}

impl<T: LleNum, L: ConstOp<T>> ConstOp<T> for Option<L> {
    fn get_value(&self, step: Step, pos: usize) -> Complex<T> {
        self.as_ref().map_or_else(zero, |x| x.get_value(step, pos))
    }
    fn skip(&self) -> bool {
        match self {
            Some(x) => x.skip(),
            None => true,
        }
    }
}

impl<T: LleNum, F: Fn(Step, usize) -> Complex<T>> ConstOp<T> for F {
    fn get_value(&self, step: Step, pos: usize) -> Complex<T> {
        self(step, pos)
    }
}

impl<T: LleNum> ConstOp<T> for fn(usize) -> Complex<T> {
    fn get_value(&self, _step: Step, pos: usize) -> Complex<T> {
        self(pos)
    }
}
