use std::{f64::consts::PI, hint::black_box};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use iterator_ilp::IteratorILP;
use lle::{num_complex::Complex64, num_traits::Zero, shift_freq, LinearOp, Step};

fn add_random(intensity: f64, sigma: f64, state: &mut [Complex64], seed: impl Into<Option<u64>>) {
    use rand::Rng;
    if let Some(seed) = seed.into() {
        use rand::SeedableRng;
        let mut rand = rand::rngs::StdRng::seed_from_u64(seed);
        state.iter_mut().for_each(|x| {
            *x += (Complex64::i() * rand.random::<f64>() * 2. * PI).exp()
                * (-(rand.random::<f64>() / sigma).powi(2) / 2.).exp()
                / ((2. * PI).sqrt() * sigma)
                * intensity
        })
    } else {
        let mut rand = rand::rng();
        state.iter_mut().for_each(|x| {
            *x += (Complex64::i() * rand.random::<f64>() * 2. * PI).exp()
                * (-(rand.random::<f64>() / sigma).powi(2) / 2.).exp()
                / ((2. * PI).sqrt() * sigma)
                * intensity
        })
    }
}

fn get_init(size: usize) -> Vec<Complex64> {
    let mut init = vec![Complex64::zero(); size];
    add_random(1e-3, 1e-5, init.as_mut_slice(), 20241020);
    init
}

const LINEAR_OP: (u32, Complex64) = (2, Complex64 { re: 1., im: 0. });
const CUR_STEP: Step = 1000;
const ILP_STREAM: usize = 8;
const STEP_DIST: f64 = 1e-4;

fn fold_ilp(state_freq: &mut [Complex64]) {
    shift_freq(state_freq).fold_ilp::<{ ILP_STREAM }, _>(
        || (),
        |_, (f, x)| {
            *x *= (LINEAR_OP.get_value(CUR_STEP, f) * STEP_DIST).exp();
        },
        |_, _| (),
    );
}

fn for_each(state_freq: &mut [Complex64]) {
    shift_freq(state_freq).for_each(|(f, x)| {
        *x *= (LINEAR_OP.get_value(CUR_STEP, f) * STEP_DIST).exp();
    });
}

fn manual_expand(state_freq: &mut [Complex64]) {
    let mut iter = shift_freq(state_freq);
    loop {
        let a = match iter.next() {
            Some(value) => value,
            None => break,
        };
        let b = match iter.next() {
            Some(value) => value,
            None => break,
        };
        let c = match iter.next() {
            Some(value) => value,
            None => break,
        };
        let d = match iter.next() {
            Some(value) => value,
            None => break,
        };
        *a.1 *= (LINEAR_OP.get_value(CUR_STEP, a.0) * STEP_DIST).exp();
        *b.1 *= (LINEAR_OP.get_value(CUR_STEP, b.0) * STEP_DIST).exp();
        *c.1 *= (LINEAR_OP.get_value(CUR_STEP, c.0) * STEP_DIST).exp();
        *d.1 *= (LINEAR_OP.get_value(CUR_STEP, d.0) * STEP_DIST).exp();
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterator linear");
    for i in 0..6 {
        let size = 128 * 2usize.pow(i);
        let init = get_init(size);
        group.bench_with_input(BenchmarkId::new("fold_ilp", size), &init, |b, s| {
            let mut s = s.clone();
            b.iter(|| fold_ilp(black_box(&mut s)))
        });
        group.bench_with_input(BenchmarkId::new("for_each", size), &init, |b, s| {
            let mut s = s.clone();
            b.iter(|| for_each(black_box(&mut s)))
        });
        group.bench_with_input(BenchmarkId::new("manual", size), &init, |b, s| {
            let mut s = s.clone();
            b.iter(|| manual_expand(black_box(&mut s)))
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
