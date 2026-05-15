use num_complex::Complex64;

use crate::foundation::math::cosine_similarity;
use crate::types::{AssociatedItem, HologramFragment};

pub struct SimilarityMatcher {
    threshold: f64,
}

impl SimilarityMatcher {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    pub fn find_similar(
        &self,
        query: &HologramFragment,
        candidates: &[HologramFragment],
        top_k: usize,
    ) -> Vec<AssociatedItem> {
        let query_freq: Vec<Complex64> = query.frequency_domain.iter().copied().collect();

        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let cand_freq: Vec<Complex64> = candidate.frequency_domain.iter().copied().collect();
                let sim = cosine_similarity(&query_freq, &cand_freq);
                (idx, sim)
            })
            .filter(|(_, sim)| *sim >= self.threshold)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(idx, sim)| AssociatedItem {
                fragment_id: candidates[idx].id,
                similarity: sim,
                metadata: candidates[idx].metadata.clone(),
            })
            .collect()
    }

    pub fn similarity(&self, a: &HologramFragment, b: &HologramFragment) -> f64 {
        let a_freq: Vec<Complex64> = a.frequency_domain.iter().copied().collect();
        let b_freq: Vec<Complex64> = b.frequency_domain.iter().copied().collect();
        cosine_similarity(&a_freq, &b_freq)
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl Default for SimilarityMatcher {
    fn default() -> Self {
        Self::new(0.3)
    }
}
