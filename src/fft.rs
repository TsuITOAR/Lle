use std::sync::Arc;

use num_complex::Complex;
use rustfft::{num_traits::Zero, FftPlanner};

use crate::LleNum;

pub trait FftSource<T: LleNum> {
    type FftProcessor;
    fn fft_len(&self) -> usize;
    fn default_fft(len: usize) -> Self::FftProcessor;
    fn scale_factor(&self) -> T {
        T::from_usize(self.fft_len()).unwrap()
    }
    fn fft_process_forward(&mut self, fft: &mut Self::FftProcessor);
    fn fft_process_inverse(&mut self, fft: &mut Self::FftProcessor);
}

impl<T: LleNum, const L: usize> FftSource<T> for [Complex<T>; L] {
    type FftProcessor = (BufferedFft<T>, BufferedFft<T>);
    fn fft_len(&self) -> usize {
        L
    }
    fn default_fft(len: usize) -> Self::FftProcessor {
        BufferedFft::new(len)
    }
    fn scale_factor(&self) -> T {
        T::from_usize(self.fft_len()).unwrap()
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
    fn fft_len(&self) -> usize {
        self.len()
    }
    fn default_fft(len: usize) -> Self::FftProcessor {
        BufferedFft::new(len)
    }
    fn scale_factor(&self) -> T {
        T::from_usize(self.len()).unwrap()
    }
    fn fft_process_forward(&mut self, fft: &mut Self::FftProcessor) {
        fft.0.fft_process(self);
    }
    fn fft_process_inverse(&mut self, fft: &mut Self::FftProcessor) {
        fft.1.fft_process(self);
    }
}

impl<T: LleNum> FftSource<T> for [Complex<T>] {
    type FftProcessor = (BufferedFft<T>, BufferedFft<T>);
    fn fft_len(&self) -> usize {
        self.len()
    }
    fn default_fft(len: usize) -> Self::FftProcessor {
        BufferedFft::new(len)
    }
    fn scale_factor(&self) -> T {
        T::from_usize(self.fft_len()).unwrap()
    }
    fn fft_process_forward(&mut self, fft: &mut Self::FftProcessor) {
        fft.0.fft_process(self);
    }
    fn fft_process_inverse(&mut self, fft: &mut Self::FftProcessor) {
        fft.1.fft_process(self);
    }
}

impl<T: LleNum, S: FftSource<T>> FftSource<T> for &mut S {
    type FftProcessor = S::FftProcessor;
    fn fft_len(&self) -> usize {
        S::fft_len(*self)
    }
    fn default_fft(len: usize) -> Self::FftProcessor {
        S::default_fft(len)
    }
    fn scale_factor(&self) -> T {
        S::scale_factor(*self)
    }
    fn fft_process_forward(&mut self, fft: &mut Self::FftProcessor) {
        S::fft_process_forward(*self, fft)
    }
    fn fft_process_inverse(&mut self, fft: &mut Self::FftProcessor) {
        S::fft_process_inverse(*self, fft)
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
