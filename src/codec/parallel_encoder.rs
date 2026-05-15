use num_complex::Complex64;
use ndarray::Array2;
use rayon::prelude::*;

use crate::foundation::config::EncodingConfig;
use crate::foundation::math::{FourierTransformer, next_power_of_two};
use crate::types::{FragmentId, FragmentMeta, HologramFragment, PhaseKey};

pub struct ParallelEncoder {
    config: EncodingConfig,
}

impl ParallelEncoder {
    pub fn new(config: EncodingConfig) -> Self {
        Self { config }
    }

    pub fn encode(&self, data: &[f64]) -> crate::codec::fourier_encoder::EncodeResult {
        let window_size = self.config.fft_window_size;
        let overlap = (window_size as f64 * self.config.overlap_ratio) as usize;
        let step = window_size - overlap;

        let n = data.len();
        if n == 0 {
            return crate::codec::fourier_encoder::EncodeResult {
                fragments: Vec::new(),
                source_hash: 0,
            };
        }

        let padded_len = next_power_of_two(n.max(window_size));
        let mut padded = vec![0.0f64; padded_len];
        padded[..n].copy_from_slice(data);

        let num_windows = if padded_len <= step { 1 } else { (padded_len - 1) / step + 1 };
        let source_hash = compute_hash(data);

        let windows: Vec<(usize, Vec<f64>)> = (0..num_windows)
            .into_par_iter()
            .map(|win_idx| {
                let start = win_idx * step;
                let end = (start + window_size).min(padded_len);
                let mut window = vec![0.0; window_size];
                window[..end - start].copy_from_slice(&padded[start..end]);
                (win_idx, window)
            })
            .collect();

        let fragments: Vec<HologramFragment> = windows
            .into_par_iter()
            .enumerate()
            .map(|(seq, (win_idx, window))| {
                let mut transformer = FourierTransformer::new();
                let window = apply_hann_window(&window);
                let freq = transformer.forward(&window);

                let phase_key = if self.config.phase_modulation {
                    PhaseKey::random(window_size)
                } else {
                    PhaseKey::zero(window_size)
                };

                let freq_modulated: Vec<Complex64> = if self.config.phase_modulation {
                    freq.iter()
                        .zip(phase_key.phases.iter())
                        .map(|(f, &p)| f * Complex64::new(p.cos(), p.sin()))
                        .collect()
                } else {
                    freq
                };

                let cols = freq_modulated.len();
                let frequency_domain = Array2::from_shape_vec((1, cols), freq_modulated)
                    .expect("频域数组形状无效");

                HologramFragment {
                    id: (seq + 1) as FragmentId,
                    frequency_domain,
                    phase_key,
                    redundancy_level: self.config.redundancy_level,
                    metadata: FragmentMeta::new(source_hash, num_windows as u32, win_idx as u32),
                }
            })
            .collect();

        crate::codec::fourier_encoder::EncodeResult {
            fragments,
            source_hash,
        }
    }
}

fn apply_hann_window(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    data.iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = if n <= 1 {
                1.0
            } else {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
            };
            x * w
        })
        .collect()
}

fn compute_hash(data: &[f64]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &val in data {
        let bytes = val.to_le_bytes();
        for &byte in &bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}
