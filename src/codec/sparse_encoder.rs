use num_complex::Complex64;
use ndarray::Array2;

use crate::types::{FragmentId, FragmentMeta, HologramFragment, PhaseKey};

pub struct SparseEncoder {
    top_k_ratio: f64,
}

pub struct SparseFragment {
    pub id: FragmentId,
    pub indices: Vec<usize>,
    pub coefficients: Vec<Complex64>,
    pub original_len: usize,
    pub metadata: FragmentMeta,
}

impl SparseEncoder {
    pub fn new(top_k_ratio: f64) -> Self {
        Self {
            top_k_ratio: top_k_ratio.clamp(0.01, 1.0),
        }
    }

    pub fn sparsify(&self, fragment: &HologramFragment) -> SparseFragment {
        let freq: Vec<Complex64> = fragment.frequency_domain.iter().copied().collect();
        let total = freq.len();

        let keep_count = ((total as f64) * self.top_k_ratio).ceil() as usize;
        let keep_count = keep_count.max(1).min(total);

        let mut indexed: Vec<(usize, f64)> = freq
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.norm_sqr()))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(keep_count);
        indexed.sort_by_key(|a| a.0);

        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let coefficients: Vec<Complex64> = indices.iter().map(|&i| freq[i]).collect();

        SparseFragment {
            id: fragment.id,
            indices,
            coefficients,
            original_len: total,
            metadata: fragment.metadata.clone(),
        }
    }

    pub fn densify(&self, sparse: &SparseFragment) -> HologramFragment {
        let mut freq = vec![Complex64::new(0.0, 0.0); sparse.original_len];
        for (k, &idx) in sparse.indices.iter().enumerate() {
            if idx < sparse.original_len {
                freq[idx] = sparse.coefficients[k];
            }
        }

        let cols = sparse.original_len;
        HologramFragment {
            id: sparse.id,
            frequency_domain: Array2::from_shape_vec((1, cols), freq)
                .expect("频域数组形状无效"),
            phase_key: PhaseKey::zero(cols),
            redundancy_level: 2,
            metadata: sparse.metadata.clone(),
        }
    }

    pub fn sparsify_batch(&self, fragments: &[HologramFragment]) -> Vec<SparseFragment> {
        fragments.iter().map(|f| self.sparsify(f)).collect()
    }

    pub fn densify_batch(&self, sparse_fragments: &[SparseFragment]) -> Vec<HologramFragment> {
        sparse_fragments.iter().map(|s| self.densify(s)).collect()
    }

    pub fn energy_analysis(&self, fragment: &HologramFragment) -> EnergyReport {
        let freq: Vec<Complex64> = fragment.frequency_domain.iter().copied().collect();
        let total_energy: f64 = freq.iter().map(|c| c.norm_sqr()).sum();

        if total_energy == 0.0 {
            return EnergyReport {
                total_energy: 0.0,
                top_10pct_energy: 0.0,
                concentration_ratio: 0.0,
                effective_dimension: 0,
                total_coefficients: freq.len(),
            };
        }

        let mut magnitudes: Vec<f64> = freq.iter().map(|c| c.norm_sqr()).collect();
        magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let top_count = ((freq.len() as f64) * 0.1).ceil() as usize;
        let top_10pct_energy: f64 = magnitudes.iter().take(top_count).sum();

        let mut cumsum = 0.0;
        let mut eff_dim = 0;
        for &m in &magnitudes {
            cumsum += m;
            eff_dim += 1;
            if cumsum >= total_energy * 0.99 {
                break;
            }
        }

        EnergyReport {
            total_energy,
            top_10pct_energy,
            concentration_ratio: top_10pct_energy / total_energy,
            effective_dimension: eff_dim,
            total_coefficients: freq.len(),
        }
    }
}

pub struct EnergyReport {
    pub total_energy: f64,
    pub top_10pct_energy: f64,
    pub concentration_ratio: f64,
    pub effective_dimension: usize,
    pub total_coefficients: usize,
}

impl std::fmt::Display for EnergyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "总能量={:.2e}, Top10%能量={:.2e}, 集中度={:.2}%, 有效维度={}/{}",
            self.total_energy,
            self.top_10pct_energy,
            self.concentration_ratio,
            self.effective_dimension,
            self.total_coefficients,
        )
    }
}
