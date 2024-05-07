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
        let coup_nonlin1 = couple.nonlinear(state1, cur_step1);
        let coup_constant1 = couple.constant(state1, cur_step1);
        let len1 = state1.len();
        let coup_linear2_r = couple.linear_f(state1, cur_step1);

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
        let coup_nonlin2 = couple.nonlinear(state2, cur_step2);
        let coup_constant2 = couple.constant(state2, cur_step2);
        let len2 = state2.len();
        let coup_linear1_r = couple.linear_f(state2, cur_step2);

        //####################################################
        // There are situations that the linear term is not shown,
        // when couple doesn't have linear term and the component doesn't have linear term
        // but it's very uncommon, so I will just ignore it
        let fft1 = fft1.get_or_insert_with(|| BufferedFft::new(len1));
        fft1.0.process(state1);
        let coup_linear1_f = couple.linear_f(state2, cur_step2);
        apply_linear_freq(
            state1,
            &linear1.by_ref().add(coup_linear1_r.add(coup_linear1_f)),
            len1,
            *step_dist1,
            cur_step1,
        );

        //####################################################
        let fft2 = fft2.get_or_insert_with(|| BufferedFft::new(len2));
        fft2.0.process(state2);
        let coup_linear2_f = couple.linear_r(state1, cur_step1);
        apply_linear_freq(
            state2,
            &linear2.by_ref().add(coup_linear2_r.add(coup_linear2_f)),
            len2,
            *step_dist2,
            cur_step2,
        );

        //####################################################

        couple.mix_freq(state1, state2);

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
            (None, Some(mut n)) => apply_nonlinear(state1, &mut n, len1, *step_dist1, cur_step1),
            (Some(n), None) => apply_nonlinear(state1, n, len1, *step_dist1, cur_step1),
            (Some(n), Some(cn)) => apply_nonlinear(
                state1,
                &mut n.by_mut().add(cn),
                len1,
                *step_dist1,
                cur_step1,
            ),
        }

        match (nonlin2, coup_nonlin2) {
            (None, None) => (),
            (None, Some(mut n)) => apply_nonlinear(state2, &mut n, len2, *step_dist2, cur_step2),
            (Some(n), None) => apply_nonlinear(state2, n, len2, *step_dist2, cur_step2),
            (Some(n), Some(cn)) => apply_nonlinear(
                state2,
                &mut n.by_mut().add(cn),
                len2,
                *step_dist2,
                cur_step2,
            ),
        }

        couple.mix(state1, state2);

        *cur_step += 1;
    }

    fn state(&self) -> &[Complex<T>] {
        self.component1.state()
    }

    fn state_mut(&mut self) -> &mut [Complex<T>] {
        self.component1.state_mut()
    }
}

// !WARN:this function will scale every element 'len' times due to fft
fn apply_linear_freq<T: LleNum, L: LinearOp<T>>(
    state_freq: &mut [Complex<T>],
    linear: &L,
    len: usize,
    step_dist: T,
    cur_step: Step,
) {
    let split_pos = (len + 1) / 2; //for odd situations, need to shift (len+1)/2..len, for evens, len/2..len;
    let (pos_freq, neg_freq) = state_freq.split_at_mut(split_pos);
    neg_freq
        .iter_mut()
        .chain(pos_freq.iter_mut())
        .enumerate()
        .for_each(|x| {
            *x.1 *= (linear.get_value(cur_step, x.0 as i32 - (len - split_pos) as i32) * step_dist)
                .exp()
        });
}

fn apply_nonlinear<T: LleNum, NL: NonLinearOp<T>>(
    state: &mut [Complex<T>],
    nonlinear: &mut NL,
    len: usize,
    step_dist: T,
    cur_step: Step,
) {
    let mut buf = vec![Complex::zero(); len];

    nonlinear.get_value(cur_step, state.as_ref(), &mut buf);
    state
        .iter_mut()
        .zip(buf.iter())
        .for_each(|x| *x.0 *= (x.1 * step_dist).exp())
}
