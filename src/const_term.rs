use crate::{LleNum, Marker, NoneOp, Step};
use num_complex::Complex;
use rustfft::num_traits::{zero, Zero};
use std::marker::PhantomData;

pub trait ConstOp<T: LleNum>: Sized + Marker {
    fn get_value(&self, cur_step: Step, pos: usize, state: &[Complex<T>]) -> Complex<T>;

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

    fn by_ref_const_op(&'_ self) -> ConstOpRef<'_, Self> {
        ConstOpRef { op: self }
    }

    fn get_value_array(&self, cur_step: Step, state: &[Complex<T>]) -> Vec<Complex<T>> {
        let len = state.len();
        (0..len)
            .map(|x| self.get_value(cur_step, x, state))
            .collect()
    }

    fn fill_value_array(&self, cur_step: Step, state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.get_value(cur_step, i, state);
        });
    }

    fn apply_const_op(&self, state: &mut [Complex<T>], cur_step: Step, step_dist: T) {
        let value = self.get_value_array(cur_step, state);
        state.iter_mut().zip(value).for_each(|(x, v)| {
            *x += v * step_dist;
        });
    }
    fn skip(&self) -> bool {
        false
    }
}

macro_rules! CompoundConst {
    ($name:ident<$g1:ident, $g2:ident>,$op:tt,$op1:tt) => {
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        pub struct $name<T:LleNum,$g1:ConstOp<T>,$g2:ConstOp<T>> {
            pub op1:$g1,
            pub op2:$g2,
            ph:PhantomData<T>
        }
        impl<T:LleNum,$g1:ConstOp<T>,$g2:ConstOp<T>> ConstOp<T> for $name<T,$g1,$g2> {
            #[inline]
            fn get_value(&self, step: Step, pos: usize,  state: &[Complex<T>])->Complex<T>{
                self.op1.get_value(step,pos,state) $op self.op2.get_value(step,pos,state)
            }


            fn get_value_array(&self, cur_step: Step, state: &[Complex<T>]) -> Vec<Complex<T>> {
                let mut array1=self.op1.get_value_array(cur_step,state);
                let array2=self.op2.get_value_array(cur_step,state);
                array1.iter_mut().zip(array2.iter()).for_each(|(x,y)|{
                    *x $op1 *y;
                });
                array1
            }

            fn fill_value_array(&self, cur_step: Step, state: &[Complex<T>], dst: &mut [Complex<T>]) {
                self.op1.fill_value_array(cur_step,state,dst);
                let array2=self.op2.get_value_array(cur_step,state);
                dst.iter_mut().zip(array2.iter()).for_each(|(x,y)|{
                    *x $op1 *y;
                });
            }

            fn skip(&self)->bool{
                self.op1.skip() && self.op2.skip()
            }
        }

        impl<T:LleNum,$g1:StaticConstOp<T>,$g2:StaticConstOp<T>> StaticConstOp<T> for $name<T,$g1,$g2> {}
    };
}

CompoundConst!( ConstOpAdd<P1, P2>,+,+=);
CompoundConst!( ConstOpSub<P1, P2>,-,-=);
CompoundConst!( ConstOpMul<P1, P2>,*,*=);
CompoundConst!( ConstOpDiv<P1, P2>,/,/=);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstOpCached<T: LleNum> {
    cache: Vec<Complex<T>>,
}

impl<T: LleNum> ConstOpCached<T> {
    pub fn new<L: ConstOp<T>>(op: L, len: usize) -> Self {
        Self {
            cache: op.get_value_array(Step::default(), &vec![Complex::zero(); len]),
        }
    }

    pub fn get_cache(&self) -> &[Complex<T>] {
        &self.cache
    }
}

impl<T: LleNum> ConstOp<T> for ConstOpCached<T> {
    fn get_value(&self, _cur_step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self.cache[pos]
    }
    fn fill_value_array(&self, _cur_step: Step, _state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.copy_from_slice(&self.cache);
    }
    fn get_value_array(&self, _cur_step: Step, _state: &[Complex<T>]) -> Vec<Complex<T>> {
        self.cache.clone()
    }
    fn apply_const_op(&self, state: &mut [Complex<T>], _cur_step: Step, step_dist: T) {
        state.iter_mut().zip(self.cache.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

#[derive(Debug, Clone)]
pub struct ConstOpRef<'a, L> {
    op: &'a L,
}

impl<T: LleNum, L: ConstOp<T>> ConstOp<T> for ConstOpRef<'_, L> {
    fn get_value(&self, cur_step: Step, pos: usize, state: &[Complex<T>]) -> Complex<T> {
        self.op.get_value(cur_step, pos, state)
    }
    fn fill_value_array(&self, cur_step: Step, state: &[Complex<T>], dst: &mut [Complex<T>]) {
        self.op.fill_value_array(cur_step, state, dst);
    }
    fn get_value_array(&self, cur_step: Step, state: &[Complex<T>]) -> Vec<Complex<T>> {
        self.op.get_value_array(cur_step, state)
    }
    fn apply_const_op(&self, state: &mut [Complex<T>], cur_step: Step, step_dist: T) {
        self.op.apply_const_op(state, cur_step, step_dist);
    }
    fn skip(&self) -> bool {
        self.op.skip()
    }
}

impl<T: LleNum> ConstOp<T> for Complex<T> {
    fn get_value(&self, _cur_step: Step, _pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        *self
    }
}

impl ConstOp<f64> for f64 {
    fn get_value(&self, _cur_step: Step, _pos: usize, _state: &[Complex<f64>]) -> Complex<f64> {
        Complex::new(*self, 0.)
    }
}

impl ConstOp<f32> for f32 {
    fn get_value(&self, _cur_step: Step, _pos: usize, _state: &[Complex<f32>]) -> Complex<f32> {
        Complex::new(*self, 0.)
    }
}

impl<T: LleNum> ConstOp<T> for Vec<Complex<T>> {
    fn get_value(&self, _cur_step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _cur_step: Step, _state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self);
    }
    fn get_value_array(&self, _cur_step: Step, _state: &[Complex<T>]) -> Vec<Complex<T>> {
        self.clone()
    }
    fn apply_const_op(&self, state: &mut [Complex<T>], _cur_step: Step, step_dist: T) {
        state.iter_mut().zip(self).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum> ConstOp<T> for &Vec<Complex<T>> {
    fn get_value(&self, _step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, _state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self);
    }
    fn get_value_array(&self, _step: Step, _state: &[Complex<T>]) -> Vec<Complex<T>> {
        (*self).clone()
    }
    fn apply_const_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

/* impl<T: LleNum> ConstOp<T> for [Complex<T>] {
    fn get_value(&self, _step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, _state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self);
    }
    fn get_value_array(&self, _step: Step, _state: &[Complex<T>]) -> Vec<Complex<T>> {
        self.to_vec()
    }
    fn apply_const_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
} */

impl<T: LleNum> ConstOp<T> for &[Complex<T>] {
    fn get_value(&self, _step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, _state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self);
    }
    fn get_value_array(&self, _step: Step, _state: &[Complex<T>]) -> Vec<Complex<T>> {
        self.to_vec()
    }
    fn apply_const_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum, const L: usize> ConstOp<T> for [Complex<T>; L] {
    fn get_value(&self, _step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, _state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self.as_slice());
    }
    fn get_value_array(&self, _step: Step, _state: &[Complex<T>]) -> Vec<Complex<T>> {
        self.to_vec()
    }
    fn apply_const_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum, const L: usize> ConstOp<T> for &[Complex<T>; L] {
    fn get_value(&self, _step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self[pos]
    }
    fn fill_value_array(&self, _step: Step, _state: &[Complex<T>], dst: &mut [Complex<T>]) {
        dst.copy_from_slice(self.as_slice());
    }
    fn get_value_array(&self, _step: Step, _state: &[Complex<T>]) -> Vec<Complex<T>> {
        self.to_vec()
    }
    fn apply_const_op(&self, state: &mut [Complex<T>], _step: Step, step_dist: T) {
        state.iter_mut().zip(self.iter()).for_each(|(x, y)| {
            *x += y * step_dist;
        });
    }
}

impl<T: LleNum> ConstOp<T> for NoneOp<T> {
    fn get_value(&self, _step: Step, _pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        Complex::zero()
    }
    fn skip(&self) -> bool {
        true
    }
}

impl<T: LleNum, L: ConstOp<T>> ConstOp<T> for Option<L> {
    fn get_value(&self, step: Step, pos: usize, state: &[Complex<T>]) -> Complex<T> {
        self.as_ref()
            .map_or_else(zero, |x| x.get_value(step, pos, state))
    }
    fn skip(&self) -> bool {
        match self {
            Some(x) => x.skip(),
            None => true,
        }
    }
}

impl<T: LleNum, F: Fn(Step, usize) -> Complex<T> + Marker> ConstOp<T> for F {
    fn get_value(&self, step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self(step, pos)
    }
}

impl<T: LleNum> ConstOp<T> for fn(usize) -> Complex<T> {
    fn get_value(&self, _step: Step, pos: usize, _state: &[Complex<T>]) -> Complex<T> {
        self(pos)
    }
}

/// Impl for all but SelfPump
pub trait StaticConstOp<T: LleNum>: ConstOp<T> {
    fn cached_const_op(self, len: usize) -> ConstOpCached<T>
    where
        Self: Sized,
    {
        ConstOpCached::new(self, len)
    }
}

impl<T: LleNum> StaticConstOp<T> for ConstOpCached<T> {
    fn cached_const_op(self, _len: usize) -> ConstOpCached<T>
    where
        Self: Sized,
    {
        unreachable!("ConstOpCached should not be used with cached_const_op")
    }
}

impl<T: LleNum, L: ConstOp<T>> StaticConstOp<T> for ConstOpRef<'_, L> {}
impl<T: LleNum> StaticConstOp<T> for NoneOp<T> {}
impl<T: LleNum, L: StaticConstOp<T>> StaticConstOp<T> for Option<L> {}

impl<T: LleNum> StaticConstOp<T> for Complex<T> {}
impl StaticConstOp<f64> for f64 {}
impl StaticConstOp<f32> for f32 {}
impl<T: LleNum> StaticConstOp<T> for Vec<Complex<T>> {}
impl<T: LleNum> StaticConstOp<T> for &Vec<Complex<T>> {}
// impl<T: LleNum> StaticConstOp<T> for [Complex<T>] {}
impl<T: LleNum> StaticConstOp<T> for &[Complex<T>] {}
impl<T: LleNum, const L: usize> StaticConstOp<T> for [Complex<T>; L] {}
impl<T: LleNum, const L: usize> StaticConstOp<T> for &[Complex<T>; L] {}

impl<T: LleNum> StaticConstOp<T> for fn(Step, usize) -> Complex<T> {}
impl<T: LleNum> StaticConstOp<T> for fn(usize) -> Complex<T> {}
