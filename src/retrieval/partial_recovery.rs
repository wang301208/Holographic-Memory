use crate::codec::redundancy_weaver::RedundancyWeaver;
use crate::types::{HologramFragment, IntegrityReport, RetrievalResult};

pub struct PartialRecoveryEngine {
    weaver: RedundancyWeaver,
}

impl PartialRecoveryEngine {
    pub fn new(redundancy_level: u8) -> Self {
        Self {
            weaver: RedundancyWeaver::new(redundancy_level),
        }
    }

    pub fn recover(
        &self,
        available_fragments: &[HologramFragment],
        total_fragment_count: u32,
    ) -> RetrievalResult {
        let recovery = self.weaver.recover_from_partial(available_fragments, total_fragment_count);

        let integrity = IntegrityReport::new(
            total_fragment_count,
            recovery.fragments.len() as u32,
        );

        let content = Vec::new();

        RetrievalResult {
            content,
            confidence: recovery.confidence,
            associations: Vec::new(),
            integrity,
        }
    }

    pub fn can_recover(
        &self,
        available_count: usize,
        total_count: u32,
    ) -> bool {
        if total_count == 0 {
            return true;
        }
        let damage_ratio = 1.0 - available_count as f64 / total_count as f64;
        damage_ratio <= 0.5
    }

    pub fn estimate_confidence(
        &self,
        available_count: usize,
        total_count: u32,
    ) -> f64 {
        if total_count == 0 {
            return 1.0;
        }
        available_count as f64 / total_count as f64
    }
}

impl Default for PartialRecoveryEngine {
    fn default() -> Self {
        Self::new(3)
    }
}
