use std::collections::HashMap;
use std::path::PathBuf;

use crate::types::{FragmentId, HologramFragment, IntegrityReport};
use crate::storage::holographic_index::HolographicIndex;
use crate::storage::lsm_index::{LsmIndex, LsmError as LsmErr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    L0 = 0,
    L1 = 1,
    L2 = 2,
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::L0 => write!(f, "L0(内存)"),
            Layer::L1 => write!(f, "L1(LSM磁盘)"),
            Layer::L2 => write!(f, "L2(mmap冷存)"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TieredError {
    #[error("L0错误: {0}")]
    L0(String),
    #[error("L1错误: {0}")]
    L1(#[from] LsmErr),
    #[error("片段未找到: {0}")]
    NotFound(FragmentId),
    #[error("无效层级: {0:?}")]
    InvalidLayer(Layer),
}

pub struct TieredConfig {
    pub l0_capacity: usize,
    pub l1_memtable_capacity: usize,
    pub l1_dir: PathBuf,
    pub promote_threshold: usize,
    pub demote_after_access: bool,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            l0_capacity: 1024,
            l1_memtable_capacity: 1024,
            l1_dir: PathBuf::from("holo_tiered_l1"),
            promote_threshold: 3,
            demote_after_access: false,
        }
    }
}

struct L0Layer {
    index: HolographicIndex,
    capacity: usize,
    access_count: HashMap<FragmentId, usize>,
    promote_threshold: usize,
}

impl L0Layer {
    fn new(capacity: usize, promote_threshold: usize) -> Self {
        Self {
            index: HolographicIndex::new(),
            capacity,
            access_count: HashMap::new(),
            promote_threshold,
        }
    }

    fn insert(&mut self, fragment: HologramFragment) -> FragmentId {
        let id = fragment.id;
        self.index.insert(fragment);
        id
    }

    fn get(&mut self, id: FragmentId) -> Option<&HologramFragment> {
        if let Some(fragment) = self.index.get(id) {
            *self.access_count.entry(id).or_insert(0) += 1;
            return Some(fragment);
        }
        None
    }

    fn get_by_source(&self, source_hash: u64) -> Vec<&HologramFragment> {
        self.index.get_by_source(source_hash)
    }

    fn remove(&mut self, id: FragmentId) -> Option<HologramFragment> {
        self.access_count.remove(&id);
        self.index.remove(id)
    }

    fn should_promote(&self, id: FragmentId) -> bool {
        self.access_count.get(&id).copied().unwrap_or(0) >= self.promote_threshold
    }

    fn is_full(&self) -> bool {
        self.index.len() >= self.capacity
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn evict_coldest(&mut self) -> Option<HologramFragment> {
        if self.index.is_empty() {
            return None;
        }
        let coldest_id = self.access_count
            .iter()
            .filter_map(|(&id, &count)| {
                if self.index.get(id).is_some() {
                    Some((id, count))
                } else {
                    None
                }
            })
            .min_by_key(|&(_, count)| count)
            .map(|(id, _)| id);

        if let Some(id) = coldest_id {
            self.access_count.remove(&id);
            self.index.remove(id)
        } else {
            let all = self.index.all_fragments();
            if let Some(first) = all.first() {
                let id = first.id;
                self.index.remove(id)
            } else {
                None
            }
        }
    }

    fn integrity_check(&self, source_hash: u64) -> IntegrityReport {
        self.index.integrity_check(source_hash)
    }

    #[allow(dead_code)]
    fn all_fragments(&self) -> Vec<&HologramFragment> {
        self.index.all_fragments()
    }

    #[allow(dead_code)]
    fn all_source_hashes(&self) -> Vec<u64> {
        self.index.all_source_hashes()
    }
}

pub struct TieredIndex {
    l0: L0Layer,
    l1: Option<LsmIndex>,
    config: TieredConfig,
}

impl TieredIndex {
    pub fn new(config: TieredConfig) -> Result<Self, TieredError> {
        let l0 = L0Layer::new(config.l0_capacity, config.promote_threshold);
        let l1 = Some(LsmIndex::open_with_capacity(&config.l1_dir, config.l1_memtable_capacity)?);
        Ok(Self { l0, l1, config })
    }

    pub fn insert(&mut self, fragment: HologramFragment) -> Result<FragmentId, TieredError> {
        if self.l0.is_full() {
            if let Some(evicted) = self.l0.evict_coldest() {
                if let Some(ref mut l1) = self.l1 {
                    l1.insert(evicted)?;
                } else {
                    self.l0.insert(evicted);
                }
            }
        }
        Ok(self.l0.insert(fragment))
    }

    pub fn insert_at(&mut self, fragment: HologramFragment, layer: Layer) -> Result<FragmentId, TieredError> {
        match layer {
            Layer::L0 => Ok(self.l0.insert(fragment)),
            Layer::L1 => {
                if let Some(ref mut l1) = self.l1 {
                    Ok(l1.insert(fragment)?)
                } else {
                    Err(TieredError::L1(LsmErr::DirectoryNotFound(self.config.l1_dir.clone())))
                }
            }
            Layer::L2 => Err(TieredError::InvalidLayer(Layer::L2)),
        }
    }

    pub fn get(&mut self, id: FragmentId) -> Result<Option<HologramFragment>, TieredError> {
        if let Some(fragment) = self.l0.get(id) {
            return Ok(Some(fragment.clone()));
        }

        if let Some(ref l1) = self.l1 {
            if let Some(fragment) = l1.get(id)? {
                if self.l0.should_promote(id) && !self.l0.is_full() {
                    self.l0.insert(fragment.clone());
                }
                return Ok(Some(fragment));
            }
        }

        Ok(None)
    }

    pub fn get_by_source(&self, source_hash: u64) -> Result<Vec<HologramFragment>, TieredError> {
        let mut results: HashMap<FragmentId, HologramFragment> = HashMap::new();

        for fragment in self.l0.get_by_source(source_hash) {
            results.insert(fragment.id, fragment.clone());
        }

        if let Some(ref l1) = self.l1 {
            for fragment in l1.get_by_source(source_hash)? {
                results.entry(fragment.id).or_insert(fragment);
            }
        }

        Ok(results.into_values().collect())
    }

    pub fn remove(&mut self, id: FragmentId) -> Result<Option<HologramFragment>, TieredError> {
        if let Some(fragment) = self.l0.remove(id) {
            return Ok(Some(fragment));
        }
        if let Some(ref mut l1) = self.l1 {
            if let Some(fragment) = l1.remove(id)? {
                return Ok(Some(fragment));
            }
        }
        Ok(None)
    }

    pub fn locate(&mut self, id: FragmentId) -> Result<Option<Layer>, TieredError> {
        if self.l0.get(id).is_some() {
            return Ok(Some(Layer::L0));
        }
        if let Some(ref l1) = self.l1 {
            if l1.get(id)?.is_some() {
                return Ok(Some(Layer::L1));
            }
        }
        Ok(None)
    }

    pub fn promote(&mut self, id: FragmentId) -> Result<bool, TieredError> {
        if self.l0.get(id).is_some() {
            return Ok(false);
        }
        if let Some(ref mut l1) = self.l1 {
            if let Some(fragment) = l1.remove(id)? {
                if self.l0.is_full() {
                    if let Some(evicted) = self.l0.evict_coldest() {
                        l1.insert(evicted)?;
                    }
                }
                self.l0.insert(fragment);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn demote(&mut self, id: FragmentId) -> Result<bool, TieredError> {
        if let Some(fragment) = self.l0.remove(id) {
            if let Some(ref mut l1) = self.l1 {
                l1.insert(fragment)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn integrity_check(&self, source_hash: u64) -> Result<IntegrityReport, TieredError> {
        let l0_report = self.l0.integrity_check(source_hash);

        let l1_available = if let Some(ref l1) = self.l1 {
            l1.get_by_source(source_hash)?.len() as u32
        } else {
            0
        };

        let total_available = l0_report.fragments_available + l1_available;
        if total_available == 0 && l0_report.fragments_total == 0 {
            let all_l1 = if let Some(ref l1) = self.l1 {
                l1.get_by_source(source_hash)?.len() as u32
            } else {
                0
            };
            if all_l1 > 0 {
                return Ok(IntegrityReport::new(all_l1.max(l0_report.fragments_total), total_available));
            }
        }

        Ok(IntegrityReport::new(l0_report.fragments_total, total_available))
    }

    pub fn len(&self) -> usize {
        let l1_len = self.l1.as_ref().map(|l| l.len()).unwrap_or(0);
        self.l0.len() + l1_len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn flush(&mut self) -> Result<(), TieredError> {
        if let Some(ref mut l1) = self.l1 {
            l1.flush()?;
        }
        Ok(())
    }

    pub fn compact(&mut self) -> Result<(), TieredError> {
        if let Some(ref mut l1) = self.l1 {
            l1.compact()?;
        }
        Ok(())
    }

    pub fn stats(&self) -> TieredStats {
        let l1_stats = self.l1.as_ref().map(|l| l.stats());
        TieredStats {
            l0_entries: self.l0.len(),
            l0_capacity: self.config.l0_capacity,
            l1_stats,
            total_entries: self.len(),
        }
    }
}

impl Drop for TieredIndex {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[derive(Debug)]
pub struct TieredStats {
    pub l0_entries: usize,
    pub l0_capacity: usize,
    pub l1_stats: Option<crate::storage::lsm_index::LsmStats>,
    pub total_entries: usize,
}

impl std::fmt::Display for TieredStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "分层索引统计:")?;
        writeln!(f, "  L0: {}/{} 条目", self.l0_entries, self.l0_capacity)?;
        if let Some(ref l1) = self.l1_stats {
            write!(f, "{}", l1)?;
        } else {
            writeln!(f, "  L1: 未启用")?;
        }
        write!(f, "  总计: {} 条目", self.total_entries)
    }
}
