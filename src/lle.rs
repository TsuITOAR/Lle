use super::*;

#[derive(typed_builder::TypedBuilder)]
pub struct LleSolver<T, S, Linear = NoneOp<T>, Nonlin = NoneOp<T>, Const = NoneOp<T>>
where
    T: LleNum,
{
    pub(crate) state: S,
    #[builder(default, setter(strip_option))]
    pub linear: Option<Linear>,
    #[builder(default, setter(strip_option))]
    pub nonlin: Option<Nonlin>,
    #[builder(default, setter(strip_option))]
    pub constant: Option<Const>,
    pub step_dist: T,
    #[builder(default, setter(skip))]
    pub(crate) fft: Option<(BufferedFft<T>, BufferedFft<T>)>,
    #[builder(default)]
    pub(crate) cur_step: Step,
}

impl<T: LleNum, S: Clone, Linear: Clone, NonLin: Clone> Clone for LleSolver<T, S, Linear, NonLin> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            linear: self.linear.clone(),
            nonlin: self.nonlin.clone(),
            constant: self.constant,
            step_dist: self.step_dist,
            fft: None,
            cur_step: 0,
        }
    }
}

impl<T: LleNum, S, Linear, NonLin, Const> LleSolver<T, S, Linear, NonLin, Const> {
    pub fn linear_mut(&mut self) -> &mut Option<Linear> {
        &mut self.linear
    }
    pub fn nonlin_mut(&mut self) -> &mut Option<NonLin> {
        &mut self.nonlin
    }
    pub fn constant_mut(&mut self) -> &mut Option<Const> {
        &mut self.constant
    }
}

impl<T, S, Linear, NonLin, Const> Evolver<T> for LleSolver<T, S, Linear, NonLin, Const>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T> + Marker,
    NonLin: NonLinearOp<T>,
    Const: ConstOp<T>,
{
    fn evolve(&mut self) {
        let len = self.state().len();
        let Self {
            state,
            linear,
            nonlin,
            constant,
            step_dist,
            fft,
            cur_step,
        } = self;
        let state = state.as_mut();

        if let Some(ref mut nonlin) = nonlin {
            apply_nonlinear(state, nonlin, *step_dist, *cur_step);
        }

        if let Some(ref linear) = linear {
            let fft = fft.get_or_insert_with(|| BufferedFft::new(len));
            apply_linear(
                state,
                &linear.by_ref_linear_op(),
                fft,
                *step_dist,
                *cur_step,
            );
            if let Some(ref c) = constant {
                apply_constant_scale(state, c, T::from_usize(len).unwrap(), *cur_step, *step_dist);
            } else {
                apply_constant_scale(
                    state,
                    &Complex::<T>::zero(),
                    T::from_usize(len).unwrap(),
                    *cur_step,
                    *step_dist,
                );
            }
        } else if let Some(ref c) = constant {
            apply_constant(state, c, *cur_step, *step_dist);
        }
        *cur_step += 1;
    }
    fn state(&self) -> &[Complex<T>] {
        self.state.as_ref()
    }
    fn state_mut(&mut self) -> &mut [Complex<T>] {
        self.state.as_mut()
    }
    fn cur_step(&self) -> Step {
        self.cur_step
    }
}
