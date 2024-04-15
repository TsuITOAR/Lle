use super::*;

#[derive(typed_builder::TypedBuilder)]
pub struct CoupledLleSolver<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: LinearOp<T>,
    NonLin1: Fn(Complex<T>) -> Complex<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear2: LinearOp<T>,
    NonLin2: Fn(Complex<T>) -> Complex<T>,
{
    pub component1: LleSolver<T, S1, Linear1, NonLin1>,
    pub component2: LleSolver<T, S2, Linear2, NonLin2>,
    pub coup_coefficient: Complex<T>,
    cur_step: usize,
}

impl<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2>
    CoupledLleSolver<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: LinearOp<T>,
    NonLin1: Fn(Complex<T>) -> Complex<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear2: LinearOp<T>,
    NonLin2: Fn(Complex<T>) -> Complex<T>,
{
    pub fn new(
        comp1: LleSolver<T, S1, Linear1, NonLin1>,
        comp2: LleSolver<T, S2, Linear2, NonLin2>,
        coup: impl Into<Complex<T>>,
    ) -> Self {
        Self {
            component1: comp1,
            component2: comp2,
            coup_coefficient: coup.into(),
            cur_step: 0,
        }
    }
}

impl<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2> Evolver<T>
    for CoupledLleSolver<T, S1, Linear1, NonLin1, S2, Linear2, NonLin2>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: LinearOp<T>,
    NonLin1: Fn(Complex<T>) -> Complex<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear2: LinearOp<T>,
    NonLin2: Fn(Complex<T>) -> Complex<T>,
{
    fn evolve(&mut self) {
        let Self {
            component1,
            component2,
            coup_coefficient,
            cur_step,
        } = self;
        let mut comp1_sqr_ave = component1
            .state()
            .iter()
            .fold(T::zero(), |a, b| a + b.norm_sqr());
        comp1_sqr_ave /= T::from_usize(component1.state().len()).unwrap();
        let mut comp2_sqr_ave = component2
            .state()
            .iter()
            .fold(T::zero(), |a, b| a + b.norm_sqr());
        comp2_sqr_ave /= T::from_usize(component2.state().len()).unwrap();

        fn evolve_comp<
            T: LleNum,
            S: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
            Linear: LinearOp<T>,
            NonLin: Fn(Complex<T>) -> Complex<T>,
        >(
            s: &mut LleSolver<T, S, Linear, NonLin>,
            counter_sqr_ave: T,
        ) {
            let len = s.state().len();
            if let Some(ref mut nonlin) = s.nonlin {
                let mut buf = vec![Complex::zero(); len];

                nonlin.get_value(s.cur_step, s.state.as_ref(), &mut buf);
                s.state.as_mut().iter_mut().zip(buf.iter()).for_each(|x| {
                    *x.0 *= ((x.1 + Complex::i() * counter_sqr_ave * T::from_f64(2.).unwrap())
                        * s.step_dist)
                        .exp()
                })
            } else {
                s.state.as_mut().iter_mut().for_each(|x| {
                    *x *= (Complex::i() * counter_sqr_ave * T::from_f64(2.).unwrap() * s.step_dist)
                        .exp()
                })
            }
            if let Some(ref linear) = s.linear {
                apply_linear(
                    s.state.as_mut(),
                    linear,
                    len,
                    s.fft.get_or_insert_with(|| BufferedFft::new(len)),
                    s.step_dist,
                    s.cur_step,
                );
                s.state
                    .as_mut()
                    .iter_mut()
                    .for_each(|x| *x /= T::from_usize(len).unwrap());
            };
            if let Some(c) = s.constant {
                s.state
                    .as_mut()
                    .iter_mut()
                    .for_each(|x| *x += c * s.step_dist);
            }
        }
        evolve_comp(component1, comp2_sqr_ave);
        evolve_comp(component2, comp1_sqr_ave);

        let len1 = component1.state().len();
        let len2 = component2.state().len();
        //couple term
        let fft1 = component1.fft.get_or_insert_with(|| BufferedFft::new(len1));
        let fft2 = component2.fft.get_or_insert_with(|| BufferedFft::new(len2));

        fft1.0.process(component1.state.as_mut());
        fft2.0.process(component2.state.as_mut());
        let s1: Box<[Complex<T>]> = component1.state.as_ref().into();
        let s2: Box<[Complex<T>]> = component2.state.as_ref().into();

        component1
            .state
            .as_mut()
            .iter_mut()
            .zip(s2.iter())
            .for_each(|(x, y)| *x += *coup_coefficient * Complex::i() * y * component1.step_dist);
        component2
            .state
            .as_mut()
            .iter_mut()
            .zip(s1.iter())
            .for_each(|(x, y)| *x += *coup_coefficient * Complex::i() * y * component2.step_dist);

        fft1.1.process(component1.state.as_mut());
        fft2.1.process(component2.state.as_mut());

        component1
            .state
            .as_mut()
            .iter_mut()
            .for_each(|x| *x /= T::from_usize(len1).unwrap());
        component2
            .state
            .as_mut()
            .iter_mut()
            .for_each(|x| *x /= T::from_usize(len2).unwrap());

        *cur_step += 1;
    }

    fn state(&self) -> &[Complex<T>] {
        self.component1.state()
    }

    fn state_mut(&mut self) -> &mut [Complex<T>] {
        self.component1.state_mut()
    }
}
