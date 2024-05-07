use super::*;

#[derive(typed_builder::TypedBuilder)]
pub struct LleSolver<T, S, Linear = NoneOp<T>, Nonlin = NoneOp<T>>
where
    T: LleNum,
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

   
impl<T, S, Linear, NonLin> Evolver<T> for LleSolver<T, S, Linear, NonLin>
where
    T: LleNum,
    S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear: LinearOp<T>,
    NonLin: NonLinearOp<T>,
{
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
    fn state(&self) -> &[Complex<T>] {
        self.state.as_ref()
    }
    fn state_mut(&mut self) -> &mut [Complex<T>] {
        self.state.as_mut()
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
