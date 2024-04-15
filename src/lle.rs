use super::*;

#[derive(typed_builder::TypedBuilder)]
pub struct LleSolver<T, S, Linear = NoneOp<T>, Nonlin = NoneOp<T>>
where
    T: LleNum,
    /* S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T = T>,
    Nonlin: Fn(Complex<T>) -> Complex<T>, */
{
    pub(crate) state: S,
    #[builder(default, setter(strip_option))]
    pub linear: Option<Linear>,
    #[builder(default, setter(strip_option))]
    pub nonlin: Option<Nonlin>,
    #[builder(default, setter(strip_option))]
    pub constant: Option<Complex<T>>,
    pub step_dist: T,
    #[builder(default, setter(skip))]
    pub(crate) fft: Option<(BufferedFft<T>, BufferedFft<T>)>,
    #[builder(default)]
    pub(crate) cur_step: Step,
}

impl<T: LleNum, S, Linear, NonLin> LleSolver<T, S, Linear, NonLin> {
    pub fn linear_mut(&mut self) -> &mut Option<Linear> {
        &mut self.linear
    }
    pub fn nonlin_mut(&mut self) -> &mut Option<NonLin> {
        &mut self.nonlin
    }
    pub fn constant_mut(&mut self) -> &mut Option<Complex<T>> {
        &mut self.constant
    }
}

impl<T, S, Linear, NonLin> LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T>,
    NonLin: NonLinearOp<T>,
{
    /* pub fn new<
        L: IntoLinearOps<T, Output = Linear>,
        N: IntoNonlinOps<T, Op = NonLin>,
        C: Into<Option<Complex<T>>>,
    >(
        init: S,
        step_dist: T,
        linear: L,
        nonlin: N,
        constant: C,
    ) -> Self {
        let len = init.as_ref().len();
        Self {
            state: init,
            linear: linear.into_linear_ops(),
            nonlin: nonlin.into_nonlin_ops(),
            constant: constant.into(),
            step_dist,
            fft: Some(BufferedFft::new(len)),
            len,
            cur_step: 0,
        }
    } */
    fn state(&self) -> &[Complex<T>] {
        self.state.as_ref()
    }
    fn state_mut(&mut self) -> &mut [Complex<T>] {
        self.state.as_mut()
    }
    fn evolve(&mut self) {
        let len = self.state().len();
        if let Some(ref mut nonlin) = self.nonlin {
            let mut buf = vec![Complex::zero(); len];

            nonlin.get_value(self.cur_step, self.state.as_ref(), &mut buf);
            self.state
                .as_mut()
                .iter_mut()
                .zip(buf.iter())
                .for_each(|x| *x.0 *= (x.1 * self.step_dist).exp())
        }
        if let Some(ref linear) = self.linear {
            apply_linear(
                self.state.as_mut(),
                linear,
                len,
                self.fft.get_or_insert_with(|| BufferedFft::new(len)),
                self.step_dist,
                self.cur_step,
            );
            let c = self.constant.unwrap_or_else(|| T::zero().into());
            self.state
                .as_mut()
                .iter_mut()
                .for_each(|x| *x = *x / T::from_usize(len).unwrap() + c * self.step_dist);
        } else if let Some(c) = self.constant {
            self.state
                .as_mut()
                .iter_mut()
                .for_each(|x| *x += c * self.step_dist)
        }
        self.cur_step += 1;
    }
}
/*
impl<T, S, Linear, NonLin> LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T> + Sync,
    NonLin: Fn(Complex<T>) -> Complex<T> + Sync + Send,
{
    fn par_evolve(&mut self) {
        if let Some(ref mut nonlin) = self.nonlin {
            nonlin.par_refresh(self.state.as_ref());
            self.state
                .as_mut()
                .par_iter_mut()
                .zip(nonlin.buff().par_iter())
                .for_each(|x| *x.0 *= (x.1 * self.step_dist).exp())
        }
        if let Some(ref linear) = self.linear {
            par_apply_linear(
                self.state.as_mut(),
                linear,
                self.len,
                self.fft.get_or_insert_with(|| BufferedFft::new(self.len)),
                self.step_dist,
                self.cur_step,
            );
            let c = self.constant.unwrap_or_else(|| T::zero().into());
            self.state
                .as_mut()
                .par_iter_mut()
                .for_each(|x| *x = *x / T::from_usize(self.len).unwrap() + c * self.step_dist);
        } else if let Some(c) = self.constant {
            self.state
                .as_mut()
                .par_iter_mut()
                .for_each(|x| *x += c * self.step_dist)
        }
        self.cur_step += 1;
    }
}
 */
impl<T, S, Linear, NonLin> Evolver<T> for LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T>,
    NonLin: NonLinearOp<T>,
{
    fn evolve(&mut self) {
        LleSolver::evolve(self)
    }
    fn state(&self) -> &[Complex<T>] {
        LleSolver::state(self)
    }
    fn state_mut(&mut self) -> &mut [Complex<T>] {
        LleSolver::state_mut(self)
    }
}

/* impl<T, S, Linear, NonLin> ParEvolver<T> for LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T> + Sync,
    NonLin: Fn(Complex<T>) -> Complex<T> + Send + Sync,
{
    fn par_evolve(&mut self) {
        LleSolver::par_evolve(self)
    }
    fn state(&self) -> &[Complex<T>] {
        LleSolver::state(self)
    }
    fn state_mut(&mut self) -> &mut [Complex<T>] {
        LleSolver::state_mut(self)
    }
}
 */