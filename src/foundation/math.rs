#![allow(unsafe_code)]

use std::collections::HashMap;
use num_complex::Complex64;
use rustfft::FftPlanner;
use rustfft::Fft;

pub struct FourierTransformer {
    planner: FftPlanner<f64>,
    forward_cache: HashMap<usize, std::sync::Arc<dyn Fft<f64>>>,
    inverse_cache: HashMap<usize, std::sync::Arc<dyn Fft<f64>>>,
}

impl FourierTransformer {
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            forward_cache: HashMap::new(),
            inverse_cache: HashMap::new(),
        }
    }

    fn get_forward_plan(&mut self, len: usize) -> std::sync::Arc<dyn Fft<f64>> {
        if let Some(plan) = self.forward_cache.get(&len) {
            return plan.clone();
        }
        let plan = self.planner.plan_fft_forward(len);
        self.forward_cache.insert(len, plan.clone());
        plan
    }

    fn get_inverse_plan(&mut self, len: usize) -> std::sync::Arc<dyn Fft<f64>> {
        if let Some(plan) = self.inverse_cache.get(&len) {
            return plan.clone();
        }
        let plan = self.planner.plan_fft_inverse(len);
        self.inverse_cache.insert(len, plan.clone());
        plan
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<Complex64> {
        let len = input.len();
        let mut buffer: Vec<Complex64> = input.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let fft = self.get_forward_plan(len);
        fft.process(&mut buffer);
        buffer
    }

    pub fn inverse(&mut self, input: &[Complex64]) -> Vec<Complex64> {
        let len = input.len();
        let mut buffer = input.to_vec();
        let ifft = self.get_inverse_plan(len);
        ifft.process(&mut buffer);
        let scale = 1.0 / len as f64;
        for val in buffer.iter_mut() {
            *val *= scale;
        }
        buffer
    }

    pub fn forward_2d(&mut self, input: &ndarray::Array2<f64>) -> ndarray::Array2<Complex64> {
        let (rows, cols) = input.dim();
        let mut result = ndarray::Array2::uninit((rows, cols));

        for i in 0..rows {
            let row: Vec<f64> = input.row(i).to_vec();
            let row_freq = self.forward(&row);
            for j in 0..cols {
                result[[i, j]] = std::mem::MaybeUninit::new(row_freq[j]);
            }
        }

        let mut result = unsafe { result.assume_init() };

        let mut col_buffer: Vec<Complex64> = Vec::with_capacity(rows);
        for j in 0..cols {
            col_buffer.clear();
            for i in 0..rows {
                col_buffer.push(result[[i, j]]);
            }
            let col_freq = self.forward_complex(&col_buffer);
            for i in 0..rows {
                result[[i, j]] = col_freq[i];
            }
        }

        result
    }

    pub fn inverse_2d(&mut self, input: &ndarray::Array2<Complex64>) -> ndarray::Array2<Complex64> {
        let (rows, cols) = input.dim();
        let mut result = input.clone();

        let mut col_buffer: Vec<Complex64> = Vec::with_capacity(rows);
        for j in 0..cols {
            col_buffer.clear();
            for i in 0..rows {
                col_buffer.push(result[[i, j]]);
            }
            let col_time = self.inverse(&col_buffer);
            for i in 0..rows {
                result[[i, j]] = col_time[i];
            }
        }

        let mut row_buffer: Vec<Complex64> = Vec::with_capacity(cols);
        for i in 0..rows {
            row_buffer.clear();
            for j in 0..cols {
                row_buffer.push(result[[i, j]]);
            }
            let row_time = self.inverse(&row_buffer);
            for j in 0..cols {
                result[[i, j]] = row_time[j];
            }
        }

        result
    }

    fn forward_complex(&mut self, input: &[Complex64]) -> Vec<Complex64> {
        let len = input.len();
        let mut buffer = input.to_vec();
        let fft = self.get_forward_plan(len);
        fft.process(&mut buffer);
        buffer
    }
}

impl Default for FourierTransformer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn inner_product(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y.conj())
        .sum()
}

pub fn norm(a: &[Complex64]) -> f64 {
    a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

pub fn cosine_similarity(a: &[Complex64], b: &[Complex64]) -> f64 {
    let dot = inner_product(a, b).re;
    let norm_a = norm(a);
    let norm_b = norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v + 1
}
