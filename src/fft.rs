use std::sync::Arc;

use num_complex::Complex;
use rustfft::{num_traits::Zero, FftPlanner};

use crate::LleNum;

pub trait FftSource<T: LleNum> {
    type FftProcessor;
    fn default_fft(&self) -> Self::FftProcessor;
    fn fft_process_forward(&mut self, fft: &mut Self::FftProcessor);
    fn fft_process_inverse(&mut self, fft: &mut Self::FftProcessor);
}

impl<T: LleNum, const L: usize> FftSource<T> for [Complex<T>; L] {
    type FftProcessor = (BufferedFft<T>, BufferedFft<T>);
    fn default_fft(&self) -> Self::FftProcessor {
        BufferedFft::new(self.len())
    }
    fn fft_process_forward(&mut self, fft: &mut Self::FftProcessor) {
        fft.0.fft_process(self);
    }
    fn fft_process_inverse(&mut self, fft: &mut Self::FftProcessor) {
        fft.1.fft_process(self);
    }
}

impl<T: LleNum> FftSource<T> for Vec<Complex<T>> {
    type FftProcessor = (BufferedFft<T>, BufferedFft<T>);
    fn default_fft(&self) -> Self::FftProcessor {
        BufferedFft::new(self.len())
    }
    fn fft_process_forward(&mut self, fft: &mut Self::FftProcessor) {
        fft.0.fft_process(self);
    }
    fn fft_process_inverse(&mut self, fft: &mut Self::FftProcessor) {
        fft.1.fft_process(self);
    }
}

impl<T: LleNum> FftSource<T> for &mut [Complex<T>] {
    type FftProcessor = (BufferedFft<T>, BufferedFft<T>);
    fn default_fft(&self) -> Self::FftProcessor {
        BufferedFft::new(self.len())
    }
    fn fft_process_forward(&mut self, fft: &mut Self::FftProcessor) {
        fft.0.fft_process(self);
    }
    fn fft_process_inverse(&mut self, fft: &mut Self::FftProcessor) {
        fft.1.fft_process(self);
    }
}

pub struct BufferedFft<T: LleNum> {
    fft: Arc<dyn rustfft::Fft<T>>,
    buf: Vec<Complex<T>>,
}

impl<T: LleNum> BufferedFft<T> {
    pub fn new(len: usize) -> (Self, Self) {
        let mut f = FftPlanner::new();
        let fft1 = f.plan_fft_forward(len);
        let fft2 = f.plan_fft_inverse(len);
        (
            Self {
                buf: vec![Complex::zero(); fft1.get_inplace_scratch_len()],
                fft: fft1,
            },
            Self {
                buf: vec![Complex::zero(); fft2.get_inplace_scratch_len()],
                fft: fft2,
            },
        )
    }

    pub fn fft_process(&mut self, data: &mut [Complex<T>]) {
        #[cfg(feature = "puffin")]
        puffin::profile_function!();
        self.fft.process_with_scratch(data, &mut self.buf)
    }
}
