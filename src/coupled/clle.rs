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
    Const1 = NoneOp<T>,
    ConstFreq1 = NoneOp<T>,
    Const2 = NoneOp<T>,
    ConstFreq2 = NoneOp<T>,
    Couple = NoneOp<T>,
> where
    T: LleNum,
    S1: FftSource<T>,
    S2: FftSource<T>,
{
    pub component1: LleSolver<T, S1, Linear1, NonLin1, Const1, ConstFreq1>,
    pub component2: LleSolver<T, S2, Linear2, NonLin2, Const2, ConstFreq2>,
    pub couple: Couple,
    #[builder(default = 0, setter(skip))]
    cur_step: u32,
}

impl<
        T,
        S1,
        S2,
        Linear1,
        Linear2,
        NonLin1,
        NonLin2,
        Const1,
        ConstFreq1,
        Const2,
        ConstFreq2,
        Couple,
    >
    CoupledLleSolver<
        T,
        S1,
        S2,
        Linear1,
        Linear2,
        NonLin1,
        NonLin2,
        Const1,
        ConstFreq1,
        Const2,
        ConstFreq2,
        Couple,
    >
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]> + FftSource<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]> + FftSource<T>,
    Linear1: LinearOp<T> + Marker,
    Linear2: LinearOp<T> + Marker,
    NonLin1: NonLinearOp<T>,
    NonLin2: NonLinearOp<T>,
    Const1: ConstOp<T>,
    ConstFreq1: ConstOp<T>,
    Const2: ConstOp<T>,
    ConstFreq2: ConstOp<T>,
    Couple: CoupleOp<T>,
{
    pub fn new(
        comp1: LleSolver<T, S1, Linear1, NonLin1, Const1, ConstFreq1>,
        comp2: LleSolver<T, S2, Linear2, NonLin2, Const2, ConstFreq2>,
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

impl<
        T,
        S1,
        S2,
        Linear1,
        Linear2,
        NonLin1,
        NonLin2,
        Const1,
        ConstFreq1,
        Const2,
        ConstFreq2,
        Couple,
    > Evolver<T>
    for CoupledLleSolver<
        T,
        S1,
        S2,
        Linear1,
        Linear2,
        NonLin1,
        NonLin2,
        Const1,
        ConstFreq1,
        Const2,
        ConstFreq2,
        Couple,
    >
where
    T: LleNum,
    S1: AsMut<[Complex<T>]> + AsRef<[Complex<T>]> + FftSource<T>,
    S2: AsMut<[Complex<T>]> + AsRef<[Complex<T>]> + FftSource<T>,
    Linear1: LinearOp<T> + Marker,
    Linear2: LinearOp<T> + Marker,
    NonLin1: NonLinearOp<T>,
    NonLin2: NonLinearOp<T>,
    Const1: ConstOp<T>,
    ConstFreq1: ConstOp<T>,
    Const2: ConstOp<T>,
    ConstFreq2: ConstOp<T>,
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

        let fft1 = component1
            .fft
            .get_or_insert_with(|| S1::default_fft(component1.state.fft_len()));
        let fft2 = component2
            .fft
            .get_or_insert_with(|| S2::default_fft(component2.state.fft_len()));

        //####################################################
        let LleSolver {
            state: state1,
            linear: linear1,
            nonlin: nonlin1,
            constant: constant1,
            constant_freq: const_freq1,
            fft: _,
            step_dist: step_dist1,
            ..
        } = component1;
        let state1 = state1.as_mut();
        let ExtractedCoupleOps {
            linear: coup_linear2,
            nonlinear: coup_nonlin2,
            constant: coup_constant2,
        } = couple.extract_coup_op(state1, cur_step1);

        //####################################################
        let LleSolver {
            state: state2,
            linear: linear2,
            nonlin: nonlin2,
            constant: constant2,
            constant_freq: const_freq2,
            fft: _,
            step_dist: step_dist2,
            ..
        } = component2;
        let state2 = state2.as_mut();
        let ExtractedCoupleOps {
            linear: coup_linear1,
            nonlinear: coup_nonlin1,
            constant: coup_constant1,
        } = couple.extract_coup_op(state2, cur_step2);

        //####################################################
        // There are situations that the linear term is not shown,
        // when couple doesn't have linear term and the component doesn't have linear term
        // but it's very uncommon, so I will just ignore it

        let state1 = &mut component1.state;
        state1.fft_process_forward(fft1);
        apply_linear_freq(
            state1.as_mut(),
            &linear1.by_ref_linear_op().add_linear_op(coup_linear1),
            *step_dist1,
            cur_step1,
        );
        const_freq1.apply_const_op(state1.as_mut(), cur_step1, *step_dist1);

        //####################################################

        let state2 = &mut component2.state;
        state2.fft_process_forward(fft2);

        apply_linear_freq(
            state2.as_mut(),
            &linear2.by_ref_linear_op().add_linear_op(coup_linear2),
            *step_dist2,
            cur_step2,
        );
        const_freq2.apply_const_op(state2.as_mut(), cur_step2, *step_dist2);

        //####################################################

        mix_freq(couple, state1.as_mut(), state2.as_mut(), *step_dist1);

        //####################################################
        state1.fft_process_inverse(fft1);
        state2.fft_process_inverse(fft2);

        //####################################################
        let scale1 = state1.scale_factor();
        let scale2 = state2.scale_factor();
        let state1 = state1.as_mut();
        let state2 = state2.as_mut();

        apply_constant_scale(
            state1,
            &constant1.by_ref_const_op().add_const_op(coup_constant1),
            scale1,
            *cur_step,
            *step_dist1,
        );

        apply_constant_scale(
            state2,
            &constant2.by_ref_const_op().add_const_op(coup_constant2),
            scale2,
            *cur_step,
            *step_dist2,
        );

        match (nonlin1, coup_nonlin1) {
            (n, None) => apply_nonlinear(state1, n, *step_dist1, cur_step1),
            (n, Some(cn)) => apply_nonlinear(
                state1,
                &mut n.by_mut().add_nonlin_op(cn),
                *step_dist1,
                cur_step1,
            ),
        }

        match (nonlin2, coup_nonlin2) {
            (n, None) => apply_nonlinear(state2, n, *step_dist2, cur_step2),
            (n, Some(cn)) => apply_nonlinear(
                state2,
                &mut n.by_mut().add_nonlin_op(cn),
                *step_dist2,
                cur_step2,
            ),
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

    fn cur_step(&self) -> Step {
        self.cur_step
    }
}
