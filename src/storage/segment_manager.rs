use crate::types::{FragmentId, HologramFragment};
use crate::storage::holographic_index::HolographicIndex;

pub struct SegmentManager {
    index: HolographicIndex,
    current_segment_id: u32,
    segment_fragment_count: u32,
    max_fragments_per_segment: u32,
}

impl SegmentManager {
    pub fn new(max_fragments_per_segment: u32) -> Self {
        Self {
            index: HolographicIndex::new(),
            current_segment_id: 0,
            segment_fragment_count: 0,
            max_fragments_per_segment,
        }
    }

    pub fn add_fragment(&mut self, fragment: HologramFragment) -> FragmentId {
        if self.segment_fragment_count >= self.max_fragments_per_segment {
            self.current_segment_id += 1;
            self.segment_fragment_count = 0;
        }
        self.segment_fragment_count += 1;
        self.index.insert(fragment)
    }

    pub fn add_fragments(&mut self, fragments: Vec<HologramFragment>) -> Vec<FragmentId> {
        fragments.into_iter().map(|f| self.add_fragment(f)).collect()
    }

    pub fn get(&self, id: FragmentId) -> Option<&HologramFragment> {
        self.index.get(id)
    }

    pub fn get_by_source(&self, source_hash: u64) -> Vec<&HologramFragment> {
        self.index.get_by_source(source_hash)
    }

    pub fn current_segment(&self) -> u32 {
        self.current_segment_id
    }

    pub fn total_fragments(&self) -> usize {
        self.index.len()
    }

    pub fn index(&self) -> &HolographicIndex {
        &self.index
    }

    pub fn index_mut(&mut self) -> &mut HolographicIndex {
        &mut self.index
    }
}

impl Default for SegmentManager {
    fn default() -> Self {
        Self::new(10000)
    }
}
