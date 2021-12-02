#![cfg(not(debug_assertions))]

use std::path::PathBuf;

use function_name::named;
use lle::{Evolver, Freq, LinearOp, LleSolver, Step};
use rustfft::num_complex::{Complex, Complex64};

const ARRAY_SIZE: usize = 1024;
const MONITOR_STEP: u32 = 100;

const STEP_DIST: f64 = 8e-4;
const CONSTANT: f64 = 3.94;
const LINEAR: f64 = -0.0444;
const ALPHA_START: f64 = -5.;
const ALPHA_END: f64 = 15.;

fn get_ini() -> Vec<Complex64> {
    let mut init = vec![Complex::<f64>::new(1e-3, 0.); ARRAY_SIZE];
    init[ARRAY_SIZE / 2] = Complex64::from(1.);
    init
}

fn get_file_path<F: AsRef<str>>(f: F) -> PathBuf {
    let mut path: PathBuf = env!("CARGO_TARGET_TMPDIR").into();
    path.push(f.as_ref());
    path
}

#[test]
#[named]
fn static_linear() {
    const STEP_NUM: u32 = 25_000;
    let nonlin = |x: Complex64| Complex::i() * x.norm_sqr();
    let linear = (0, -(Complex64::i() * ALPHA_START + 1.)).add((2, -Complex64::i() * LINEAR / 2.));

    let mut s = LleSolver::new(
        get_ini(),
        STEP_DIST,
        linear,
        nonlin,
        Complex64::from(CONSTANT),
    );
    use jkplot::ColorMapVisualizer;
    let mut visualizer = ColorMapVisualizer::new(
        get_file_path(concat!(function_name!(), ".png")),
        (ARRAY_SIZE as u32 * 2, ARRAY_SIZE as u32 * 2),
    );
    let mut frame = 0;
    s.evolve_n_with_monitor(STEP_NUM, |s| {
        if frame % MONITOR_STEP == 0 {
            visualizer.push(s.iter().map(|x| x.re).collect());
        }
        frame += 1;
    });
    visualizer.draw();
}

#[test]
#[named]
fn moving_linear() {
    const STEP_NUM: u32 = 250_000;
    let nonlin = |x: Complex64| Complex::i() * x.norm_sqr();
    let alpha_step = (ALPHA_END - ALPHA_START) as f64 / STEP_NUM as f64;
    let linear =
        (|step: Step, _| -(Complex64::i() * (ALPHA_START + alpha_step * step as f64) + 1.))
            .add((2, -Complex64::i() * LINEAR / 2.));
    let mut s = LleSolver::new(
        get_ini(),
        STEP_DIST,
        linear,
        nonlin,
        Complex64::from(CONSTANT),
    );
    use jkplot::ColorMapVisualizer;
    let mut visualizer = ColorMapVisualizer::new(
        get_file_path(concat!(function_name!(), ".png")),
        (ARRAY_SIZE as u32 * 2, ARRAY_SIZE as u32 * 2),
    );
    let mut frame = 0;
    s.evolve_n_with_monitor(STEP_NUM, |s| {
        if frame % MONITOR_STEP == 0 {
            visualizer.push(s.iter().map(|x| x.re).collect());
        }
        frame += 1;
    });
    visualizer.draw();
}

#[test]
#[named]
fn step_linear() {
    const DELTA: f64 = 2.;
    const STEP_NUM: u32 = 250_000;
    const ALPHA_START: f64 = -DELTA * 2.;
    const ALPHA_END: f64 = DELTA * 5.;
    let nonlin = |x: Complex64| Complex::i() * x.norm_sqr();
    let alpha_step = (ALPHA_END - ALPHA_START) as f64 / STEP_NUM as f64;
    let linear = (|step: Step, freq: Freq| {
        -(Complex64::i() * (ALPHA_START + alpha_step * step as f64) + 1.)
            - if freq == 0 {
                Complex64::from(0.)
            } else {
                Complex64::from(DELTA)
            }
    })
    .add((2, -Complex64::i() * LINEAR / 2.));
    let mut s = LleSolver::new(
        get_ini(),
        STEP_DIST,
        linear,
        nonlin,
        Complex64::from(CONSTANT),
    );
    use jkplot::ColorMapVisualizer;
    let mut visualizer = ColorMapVisualizer::new(
        get_file_path(concat!(function_name!(), ".png")),
        (ARRAY_SIZE as u32 * 2, ARRAY_SIZE as u32 * 2),
    );
    let mut frame = 0;
    s.evolve_n_with_monitor(STEP_NUM, |s| {
        if frame % MONITOR_STEP == 0 {
            visualizer.push(s.iter().map(|x| x.re).collect());
        }
        frame += 1;
    });
    visualizer.draw();
}
