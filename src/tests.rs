use super::*;
use crate::{Evolver, LinearOp, LleSolver, NoneOp, Step};
use num_traits::zero;
use rustfft::num_complex::Complex64;
#[test]
fn linear_ops() {
    let linear1 = (1, Complex64::from(1.));
    assert_eq!(linear1.get_value(1, 1), 1. * (-Complex64::i() * 1.).powu(1));
    assert_eq!(linear1.get_value(3, 2), 1. * (-Complex64::i() * 2.).powu(1));

    let linear2 = (3, Complex64::from(2.));
    assert_eq!(linear2.get_value(1, 1), 2. * (-Complex64::i() * 1.).powu(3));
    assert_eq!(linear2.get_value(2, 3), 2. * (-Complex64::i() * 3.).powu(3));

    let linear3 = linear1.clone().add(linear2.clone());
    assert_eq!(
        linear3.get_value(1, 3),
        linear1.get_value(1, 3) + linear2.get_value(1, 3)
    );
    assert_eq!(
        linear3.get_value(5, 7),
        linear1.get_value(5, 7) + linear2.get_value(5, 7)
    );

    let linear4 = (1u32, |step: Step| Complex64::from(step as f64));
    assert_eq!(
        linear4.get_value(2, 4),
        Complex64::from(2.) * (-Complex64::i() * 4.).powu(1)
    );
    assert_eq!(
        linear4.get_value(3, 7),
        Complex64::from(3.) * (-Complex64::i() * 7.).powu(1)
    );

    let linear5 = |step: Step, _pos: i32| Complex64::from(step as f64);
    assert_eq!(linear5.get_value(1, 3), (1.).into());
    assert_eq!(linear5.get_value(6, 3), (6.).into());
}

/*  #[test]
fn create_solver() {
    LleSolver::new([Complex64::new(1., 0.); 128], 0.1, none_op(), None, None);
} */
#[test]
fn zero_nonlin_op() {
    let start = [Complex64::new(1., 0.)];
    let mut s = LleSolver::<_, _>::builder()
        .state(start.clone())
        .step_dist(0.1)
        .build();
    s.evolve_n_with_monitor(10, |x| {
        assert_eq!(start, x, "zero nonlin operation should not change state")
    });
}
#[test]
fn zero_linear_op() {
    let start = [Complex64::new(1., 0.); 128];
    let mut s = LleSolver::<_, _, _>::builder()
        .state(start.clone())
        .step_dist(0.1)
        .linear((1u32, Complex64::from(0.)).add((2, Complex64::from(0.))))
        .build();

    s.evolve_n_with_monitor(10, |x| {
        assert_eq!(start, x, "zero linear operation should not change state")
    });
}
#[test]
fn evolve_some() {
    let step_dist = 0.1;
    let mut start = [Complex64::new(1., 0.); 128];
    start[35] = (0.).into();
    let mut s = LleSolver::<_, _, _>::builder()
        .state(start.clone())
        .step_dist(0.1)
        .linear((1u32, Complex64::from(1.)).add((2, Complex64::from(1.))))
        .build();

    s.evolve();
    assert_ne!(start, s.state(),);
    let nonlin = |x: Complex64| x.sqrt();
    let mut s = LleSolver::<_, _, NoneOp<_>, _>::builder()
        .state(start.clone())
        .step_dist(0.1)
        .nonlin(nonlin)
        .build();
    s.evolve();
    assert_ne!(start, s.state(),);
    let constant = Complex64::from(1.);
    let mut s = LleSolver::<_, _>::builder()
        .state(start.clone())
        .step_dist(0.1)
        .constant(constant)
        .build();
    s.evolve();
    assert_eq!(
        start
            .iter()
            .map(|x| x + step_dist * constant)
            .collect::<Vec<_>>(),
        s.state(),
    );
}

#[test]
fn fft_scale_check() {
    let array = [1., 2., 3., 4., 5., 6., 7., 8.];
    let mut fft = rustfft::FftPlanner::new();
    let fft = fft.plan_fft_forward(array.len());
    let mut freq = array
        .iter()
        .map(|x| Complex::new(*x, 0.))
        .collect::<Vec<_>>();
    fft.process(&mut freq);
    let e_real = array.iter().fold(0., |t, x| t + x * x);
    let e_freq = freq.iter().fold(zero(), |t: f64, x| t + x.norm_sqr()) / array.len() as f64;
    assert_eq!(e_real, e_freq);
}

#[test]
fn check_freq_index() {
    fn freq_at1(len: usize, i: usize) -> Freq {
        let split = (len + 1) / 2;
        if i < split {
            i as Freq
        } else {
            i as Freq - len as Freq
        }
    }
    for len in [5, 6, 7] {
        for i in [1, 2, 3, 4] {
            assert_eq!(freq_at(len, i), freq_at1(len, i));
        }
    }
}

#[test]
fn check_freq_index2() {
    fn d((i, x): (i32, &mut f32)) {
        *x *= i as f32;
    }
    fn modify1(state_freq: &mut [f32]) {
        shift_freq(state_freq).for_each(d);
    }
    fn modify2(state_freq: &mut [f32]) {
        let len = state_freq.len();
        state_freq
            .iter_mut()
            .enumerate()
            .map(|x| (freq_at(len, x.0), x.1))
            .for_each(d);
    }

    let data = [1., 2., 3., 1., 2., 5., 87., 32., 7., 0., -1.];
    let mut data1 = data.clone();
    modify1(&mut data1);
    let mut data2 = data.clone();
    modify2(&mut data2);
    assert_eq!(data1, data2);
    let data = [1., 2., 3., 2., 5., 87., 32., 7., 0., -1.];
    let mut data1 = data.clone();
    modify1(&mut data1);
    let mut data2 = data.clone();
    modify2(&mut data2);
    assert_eq!(data1, data2);
}
