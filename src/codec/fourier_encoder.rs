use num_complex::Complex64;
use ndarray::Array2;

use crate::foundation::config::EncodingConfig;
use crate::foundation::math::{FourierTransformer, next_power_of_two};
use crate::types::{FragmentId, FragmentMeta, HologramFragment, PhaseKey};

pub struct FourierEncoder {
    transformer: FourierTransformer,
    config: EncodingConfig,
    next_id: FragmentId,
    hann_cache: Vec<f64>,
}

impl FourierEncoder {
    pub fn new(config: EncodingConfig) -> Self {
        let window_size = config.fft_window_size;
        let hann_cache = precompute_hann(window_size);
        Self {
            transformer: FourierTransformer::new(),
            config,
            next_id: 1,
            hann_cache,
        }
    }

    pub fn encode(&mut self, data: &[f64]) -> EncodeResult {
        let window_size = self.config.fft_window_size;
        let overlap = (window_size as f64 * self.config.overlap_ratio) as usize;
        let step = window_size - overlap;

        let n = data.len();
        if n == 0 {
            return EncodeResult {
                fragments: Vec::new(),
                source_hash: 0,
            };
        }

        let padded_len = next_power_of_two(n.max(window_size));
        let mut padded = vec![0.0f64; padded_len];
        padded[..n].copy_from_slice(data);

        let num_windows = if padded_len <= step { 1 } else { (padded_len - 1) / step + 1 };
        let mut fragments = Vec::with_capacity(num_windows);

        let source_hash = compute_hash(data);

        for win_idx in 0..num_windows {
            let start = win_idx * step;
            let end = (start + window_size).min(padded_len);

            let mut window = vec![0.0; window_size];
            window[..end - start].copy_from_slice(&padded[start..end]);

            apply_hann_window_inplace(&mut window, &self.hann_cache);

            let freq = self.transformer.forward(&window);

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

            let fragment = HologramFragment {
                id: self.next_id,
                frequency_domain,
                phase_key,
                redundancy_level: self.config.redundancy_level,
                metadata: FragmentMeta::new(source_hash, num_windows as u32, win_idx as u32),
            };
            self.next_id += 1;
            fragments.push(fragment);
        }

        EncodeResult {
            fragments,
            source_hash,
        }
    }

    pub fn decode(&mut self, fragments: &[HologramFragment], expected_len: usize) -> Vec<f64> {
        if fragments.is_empty() {
            return Vec::new();
        }

        let window_size = self.config.fft_window_size;
        let overlap = (window_size as f64 * self.config.overlap_ratio) as usize;
        let step = window_size - overlap;

        let max_idx = fragments.iter().map(|f| f.metadata.fragment_index as usize).max().unwrap_or(0);
        let result_len = expected_len.max((max_idx + 1) * step + window_size);
        let mut result = vec![0.0f64; result_len];
        let mut weight_sum = vec![0.0f64; result_len];

        for fragment in fragments {
            let freq: Vec<Complex64> = fragment.frequency_domain.iter().copied().collect();

            let freq_restored: Vec<Complex64> = if self.config.phase_modulation && fragment.phase_key.dimension > 0 {
                freq.iter()
                    .zip(fragment.phase_key.phases.iter())
                    .map(|(f, &p)| f * Complex64::new(p.cos(), -p.sin()))
                    .collect()
            } else {
                freq
            };

            let time_data = self.transformer.inverse(&freq_restored);

            let win_idx = fragment.metadata.fragment_index as usize;
            let start = win_idx * step;

            for (i, &val) in time_data.iter().enumerate().take(window_size) {
                let idx = start + i;
                if idx < result.len() {
                    let w = self.hann_cache[i];
                    result[idx] += val.re * w;
                    weight_sum[idx] += w * w;
                }
            }
        }

        for i in 0..result.len() {
            if weight_sum[i] > 1e-10 {
                result[i] /= weight_sum[i];
            }
        }

        result.truncate(expected_len);
        result
    }
}

pub struct EncodeResult {
    pub fragments: Vec<HologramFragment>,
    pub source_hash: u64,
}

fn precompute_hann(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n.max(1)];
    }
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())).collect()
}

fn apply_hann_window_inplace(data: &mut [f64], hann: &[f64]) {
    for (d, &w) in data.iter_mut().zip(hann.iter()) {
        *d *= w;
    }
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
