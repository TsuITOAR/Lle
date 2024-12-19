use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lle::{num_complex::Complex64, num_traits::Zero, LleSolver};

fn add_random(intensity: f64, sigma: f64, state: &mut [Complex64], seed: impl Into<Option<u64>>) {
    use rand::Rng;
    if let Some(seed) = seed.into() {
        use rand::SeedableRng;
        let mut rand = rand::rngs::StdRng::seed_from_u64(seed);
        state.iter_mut().for_each(|x| {
            *x += (Complex64::i() * rand.gen::<f64>() * 2. * PI).exp()
                * (-(rand.gen::<f64>() / sigma).powi(2) / 2.).exp()
                / ((2. * PI).sqrt() * sigma)
                * intensity
        })
    } else {
        let mut rand = rand::thread_rng();
        state.iter_mut().for_each(|x| {
            *x += (Complex64::i() * rand.gen::<f64>() * 2. * PI).exp()
                * (-(rand.gen::<f64>() / sigma).powi(2) / 2.).exp()
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

const LINEAR: f64 = -0.0444;
const ALPHA_START: f64 = -5.;
const STEP_DIST: f64 = 1e-4;

use lle::{Evolver, LinearOp, Step};
fn bench_linear(size: usize) {
    const STEP: Step = 1000;
    let linear =
        (0, -(Complex64::i() * ALPHA_START + 1.)).add_linear_op((2, -Complex64::i() * LINEAR / 2.));
    let mut solver: LleSolver<f64, _, _> = LleSolver::new(get_init(size), STEP_DIST).linear(linear);
    solver.evolve_n(STEP);
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear size");
    for i in 0..8 {
        let size = 128 * 2usize.pow(i);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, s| {
            b.iter(|| bench_linear(black_box(*s)))
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
