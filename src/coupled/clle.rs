use super::*;

#[derive(typed_builder::TypedBuilder)]
pub struct CoupledLleSolver<
    T,
    S1,
    S2,
    Linear1 = NoneOp<T>,
    Linear2 = NoneOp<T>,
    NonLin1 = NoneOp<T>,
    NonLin2 = NoneOp<T>,
    Couple = NoneOp<T>,
> where
    T: LleNum,
{
    pub component1: LleSolver<T, S1, Linear1, NonLin1>,
    pub component2: LleSolver<T, S2, Linear2, NonLin2>,
    pub couple: Couple,
    #[builder(default = 0, setter(skip))]
    cur_step: u32,
}

impl<T, S1, S2, Linear1, Linear2, NonLin1, NonLin2, Couple>
    CoupledLleSolver<T, S1, S2, Linear1, Linear2, NonLin1, NonLin2, Couple>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: LinearOp<T>,
    Linear2: LinearOp<T>,
    NonLin1: NonLinearOp<T>,
    NonLin2: NonLinearOp<T>,
    Couple: CoupleOp<T>,
{
    pub fn new(
        comp1: LleSolver<T, S1, Linear1, NonLin1>,
        comp2: LleSolver<T, S2, Linear2, NonLin2>,
        couple: Couple,
    ) -> Self {
        Self {
            component1: comp1,
            component2: comp2,
            couple,
            cur_step: 0,
        }
    }
}

impl<T, S1, S2, Linear1, Linear2, NonLin1, NonLin2, Couple> Evolver<T>
    for CoupledLleSolver<T, S1, S2, Linear1, Linear2, NonLin1, NonLin2, Couple>
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]>,
    Linear1: LinearOp<T>,
    Linear2: LinearOp<T>,
    NonLin1: NonLinearOp<T>,
    NonLin2: NonLinearOp<T>,
    Couple: CoupleOp<T>,
{
    fn evolve(&mut self) {
        let Self {
            component1,
            component2,
            couple,
            cur_step,
        } = self;
        let cur_step1 = *cur_step;
        let cur_step2 = *cur_step;
        //####################################################
        let LleSolver {
            state: state1,
            linear: linear1,
            nonlin: nonlin1,
            constant: constant1,
            fft: fft1,
            step_dist: step_dist1,
            ..
        } = component1;
        let state1 = state1.as_mut();
        let ExtractedCoupleOps {
            linear: coup_linear2,
            nonlinear: coup_nonlin2,
            constant: coup_constant2,
        } = couple.extract_coup_op(state1, cur_step1);
        let len1 = state1.len();

        //####################################################
        let LleSolver {
            state: state2,
            linear: linear2,
            nonlin: nonlin2,
            constant: constant2,
            fft: fft2,
            step_dist: step_dist2,
            ..
        } = component2;
        let state2 = state2.as_mut();
        let ExtractedCoupleOps {
            linear: coup_linear1,
            nonlinear: coup_nonlin1,
            constant: coup_constant1,
        } = couple.extract_coup_op(state2, cur_step2);
        let len2 = state2.len();

        //####################################################
        // There are situations that the linear term is not shown,
        // when couple doesn't have linear term and the component doesn't have linear term
        // but it's very uncommon, so I will just ignore it
        let fft1 = fft1.get_or_insert_with(|| BufferedFft::new(len1));
        fft1.0.process(state1);
        apply_linear_freq(
            state1,
            &linear1.by_ref().add(coup_linear1),
            *step_dist1,
            cur_step1,
        );

        //####################################################
        let fft2 = fft2.get_or_insert_with(|| BufferedFft::new(len2));
        fft2.0.process(state2);

        apply_linear_freq(
            state2,
            &linear2.by_ref().add(coup_linear2),
            *step_dist2,
            cur_step2,
        );

        //####################################################

        couple.mix_freq(state1, state2, *step_dist1);

        //####################################################
        fft1.1.process(state1);
        fft2.1.process(state2);

        //####################################################
        state1.iter_mut().for_each(|x| {
            *x = *x / T::from_usize(len1).unwrap()
                + (constant1.unwrap_or_else(Complex::zero)
                    + coup_constant1.unwrap_or_else(Complex::zero))
                    * *step_dist1
        });
        state2.iter_mut().for_each(|x| {
            *x = *x / T::from_usize(len2).unwrap()
                + (constant2.unwrap_or_else(Complex::zero)
                    + coup_constant2.unwrap_or_else(Complex::zero))
                    * *step_dist2
        });

        match (nonlin1, coup_nonlin1) {
            (None, None) => (),
            (None, Some(mut n)) => apply_nonlinear(state1, &mut n, *step_dist1, cur_step1),
            (Some(n), None) => apply_nonlinear(state1, n, *step_dist1, cur_step1),
            (Some(n), Some(cn)) => {
                apply_nonlinear(state1, &mut n.by_mut().add(cn), *step_dist1, cur_step1)
            }
        }

        match (nonlin2, coup_nonlin2) {
            (None, None) => (),
            (None, Some(mut n)) => apply_nonlinear(state2, &mut n, *step_dist2, cur_step2),
            (Some(n), None) => apply_nonlinear(state2, n, *step_dist2, cur_step2),
            (Some(n), Some(cn)) => {
                apply_nonlinear(state2, &mut n.by_mut().add(cn), *step_dist2, cur_step2)
            }
        }

        mix(couple, state1, state2, *step_dist1);

        *cur_step += 1;
    }

    fn state(&self) -> &[Complex<T>] {
        self.component1.state()
    }

    fn state_mut(&mut self) -> &mut [Complex<T>] {
        self.component1.state_mut()
    }
}
