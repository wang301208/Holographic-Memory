use ndarray::Array2;
use num_complex::Complex64;

use crate::types::{FragmentId, FragmentMeta, HologramFragment, PhaseKey};

pub struct HologramFragmenter {
    fragment_size: usize,
}

impl HologramFragmenter {
    pub fn new(fragment_size: usize) -> Self {
        Self {
            fragment_size: fragment_size.max(1),
        }
    }

    pub fn fragment(&self, fragments: &[HologramFragment]) -> Vec<HologramFragment> {
        if fragments.is_empty() {
            return Vec::new();
        }

        let all_freqs: Vec<Complex64> = fragments
            .iter()
            .flat_map(|f| f.frequency_domain.iter().copied())
            .collect();

        if all_freqs.is_empty() {
            return Vec::new();
        }

        let total_len = all_freqs.len();
        let frag_size = self.fragment_size;
        let num_output = ((total_len as f64) / frag_size as f64).ceil() as usize;
        let num_output = num_output.max(1);

        let mut result = Vec::with_capacity(num_output);
        let id_counter: FragmentId = fragments.iter().map(|f| f.id).max().unwrap_or(0) + 1;

        let redundancy_level = fragments[0].redundancy_level;
        let source_hash = fragments[0].metadata.source_hash;

        for (frag_idx, id_val) in (0..num_output).zip(id_counter..) {
            let start = frag_idx * frag_size;
            let end = (start + frag_size).min(total_len);

            let mut chunk = vec![Complex64::new(0.0, 0.0); frag_size];

            for (k, slot) in chunk.iter_mut().enumerate() {
                for (j, &freq_val) in all_freqs[start..end].iter().enumerate() {
                    let global_j = start + j;
                    let phase = 2.0 * std::f64::consts::PI * (frag_idx as f64 * global_j as f64 + k as f64 * j as f64) / (num_output * frag_size) as f64;
                    *slot += freq_val * Complex64::new(phase.cos(), phase.sin());
                }
                *slot /= (end - start) as f64;
            }

            let fragment = HologramFragment {
                id: id_val,
                frequency_domain: Array2::from_shape_vec((1, frag_size), chunk)
                    .expect("分片形状无效"),
                phase_key: PhaseKey::random(frag_size),
                redundancy_level,
                metadata: FragmentMeta::new(
                    source_hash,
                    num_output as u32,
                    frag_idx as u32,
                ),
            };
            result.push(fragment);
        }

        result
    }

    pub fn defragment(&self, fragments: &[HologramFragment]) -> Vec<Complex64> {
        if fragments.is_empty() {
            return Vec::new();
        }

        let num_fragments = fragments.len();
        let frag_size = self.fragment_size;
        let total_len = num_fragments * frag_size;
        let mut result = vec![Complex64::new(0.0, 0.0); total_len];

        for (frag_idx, fragment) in fragments.iter().enumerate() {
            let freqs: Vec<Complex64> = fragment.frequency_domain.iter().copied().collect();

            for (k, &val) in freqs.iter().enumerate().take(frag_size) {
                for j in 0..frag_size {
                    let global_j = frag_idx * frag_size + j;
                    let local_j = j;
                    let phase = 2.0 * std::f64::consts::PI * (frag_idx as f64 * global_j as f64 + k as f64 * local_j as f64) / total_len as f64;
                    result[global_j] += val * Complex64::new(phase.cos(), -phase.sin());
                }
            }
        }

        let scale = num_fragments as f64;
        for val in result.iter_mut() {
            *val /= scale;
        }

        result
    }
}
