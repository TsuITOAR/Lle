use super::*;

#[derive(typed_builder::TypedBuilder)]
pub struct LleSolver<T, State, Linear = NoneOp<T>, Nonlin = NoneOp<T>, Const = NoneOp<T>>
where
    T: LleNum,
    State: FftSource<T>,
{
    pub(crate) state: State,
    pub linear: Linear,
    pub nonlin: Nonlin,
    pub constant: Const,
    pub step_dist: T,
    #[builder(default, setter(skip))]
    pub(crate) fft: Option<State::FftProcessor>,
    #[builder(default)]
    pub(crate) cur_step: Step,
}

impl<T: LleNum, State: FftSource<T>> LleSolver<T, State, NoneOp<T>, NoneOp<T>, NoneOp<T>> {
    pub fn new(init: State, step_dist: T) -> Self {
        Self {
            state: init,
            linear: NoneOp::default(),
            nonlin: NoneOp::default(),
            constant: NoneOp::default(),
            step_dist,
            fft: None,
            cur_step: 0,
        }
    }
}

impl<T: LleNum, S: Clone + FftSource<T>, Linear: Clone, NonLin: Clone> Clone
    for LleSolver<T, S, Linear, NonLin>
{
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

impl<T: LleNum, State: FftSource<T>, Linear, NonLin, Const>
    LleSolver<T, State, Linear, NonLin, Const>
{
    pub fn linear_mut(&mut self) -> &mut Linear {
        &mut self.linear
    }
    pub fn nonlin_mut(&mut self) -> &mut NonLin {
        &mut self.nonlin
    }
    pub fn constant_mut(&mut self) -> &mut Const {
        &mut self.constant
    }

    pub fn linear<L: LinearOp<T>>(self, linear: L) -> LleSolver<T, State, L, NonLin, Const> {
        LleSolver {
            state: self.state,
            linear,
            nonlin: self.nonlin,
            constant: self.constant,
            step_dist: self.step_dist,
            fft: self.fft,
            cur_step: self.cur_step,
        }
    }

    pub fn nonlin<N: NonLinearOp<T>>(self, nonlin: N) -> LleSolver<T, State, Linear, N, Const> {
        LleSolver {
            state: self.state,
            linear: self.linear,
            nonlin,
            constant: self.constant,
            step_dist: self.step_dist,
            fft: self.fft,
            cur_step: self.cur_step,
        }
    }

    pub fn constant<C: ConstOp<T>>(self, constant: C) -> LleSolver<T, State, Linear, NonLin, C> {
        LleSolver {
            state: self.state,
            linear: self.linear,
            nonlin: self.nonlin,
            constant,
            step_dist: self.step_dist,
            fft: self.fft,
            cur_step: self.cur_step,
        }
    }
}

impl<T, S, Linear, NonLin, Const> Evolver<T> for LleSolver<T, S, Linear, NonLin, Const>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]> + FftSource<T>,
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
        let state0 = state.as_mut();

        apply_nonlinear(state0, nonlin, *step_dist, *cur_step);

        let fft = fft.get_or_insert_with(|| state.default_fft());

        apply_linear(
            state,
            &linear.by_ref_linear_op(),
            fft,
            *step_dist,
            *cur_step,
        );
        
        let state0 = state.as_mut();
        apply_constant_scale(
            state0,
            constant,
            T::from_usize(len).unwrap(),
            *cur_step,
            *step_dist,
        );

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
