use std::collections::HashMap;

use crate::codec::sparse_encoder::{SparseEncoder, SparseFragment};
use crate::types::{FragmentId, HologramFragment, IntegrityReport};

pub struct SparseIndex {
    fragments: HashMap<FragmentId, SparseFragment>,
    source_groups: HashMap<u64, Vec<FragmentId>>,
    encoder: SparseEncoder,
    next_id: FragmentId,
}

impl SparseIndex {
    pub fn new(top_k_ratio: f64) -> Self {
        Self {
            fragments: HashMap::new(),
            source_groups: HashMap::new(),
            encoder: SparseEncoder::new(top_k_ratio),
            next_id: 1,
        }
    }

    pub fn insert(&mut self, fragment: HologramFragment) -> FragmentId {
        let sparse = self.encoder.sparsify(&fragment);
        self.insert_sparse(sparse)
    }

    pub fn insert_sparse(&mut self, fragment: SparseFragment) -> FragmentId {
        let id = fragment.id;
        let source_hash = fragment.metadata.source_hash;
        self.fragments.insert(id, fragment);
        self.source_groups.entry(source_hash).or_default().push(id);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        id
    }

    pub fn get(&self, id: FragmentId) -> Option<HologramFragment> {
        self.fragments.get(&id).map(|fragment| self.encoder.densify(fragment))
    }

    pub fn get_sparse(&self, id: FragmentId) -> Option<&SparseFragment> {
        self.fragments.get(&id)
    }

    pub fn get_by_source(&self, source_hash: u64) -> Vec<HologramFragment> {
        self.source_groups
            .get(&source_hash)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.get(*id))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_sparse_by_source(&self, source_hash: u64) -> Vec<&SparseFragment> {
        self.source_groups
            .get(&source_hash)
            .map(|ids| ids.iter().filter_map(|id| self.fragments.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn remove(&mut self, id: FragmentId) -> Option<HologramFragment> {
        let fragment = self.fragments.remove(&id)?;
        let source_hash = fragment.metadata.source_hash;
        if let Some(ids) = self.source_groups.get_mut(&source_hash) {
            ids.retain(|&x| x != id);
            if ids.is_empty() {
                self.source_groups.remove(&source_hash);
            }
        }
        Some(self.encoder.densify(&fragment))
    }

    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    pub fn all_fragments(&self) -> Vec<HologramFragment> {
        self.fragments
            .values()
            .map(|fragment| self.encoder.densify(fragment))
            .collect()
    }

    pub fn all_source_hashes(&self) -> Vec<u64> {
        self.source_groups.keys().copied().collect()
    }

    pub fn integrity_check(&self, source_hash: u64) -> IntegrityReport {
        let fragments = self.get_by_source(source_hash);
        let total = if let Some(first) = fragments.first() {
            first.metadata.fragment_count
        } else {
            0
        };
        IntegrityReport::new(total, fragments.len() as u32)
    }

    pub fn compression_ratio(&self, fragment: &HologramFragment) -> f64 {
        let sparse = self.encoder.sparsify(fragment);
        sparse.storage_ratio(fragment)
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new(0.3)
    }
}
