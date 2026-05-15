use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::types::{FragmentId, HologramFragment, IntegrityReport};

const DEFAULT_MEMTABLE_CAPACITY: usize = 1024;
const DEFAULT_LEVEL0_SSTABLE_LIMIT: usize = 4;
const SSTABLE_EXT: &str = ".sst";
const MANIFEST_FILE: &str = "MANIFEST";

#[derive(Debug, thiserror::Error)]
pub enum LsmError {
    #[error("IO错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("序列化错误: {0}")]
    Serialization(String),
    #[error("片段未找到: {0}")]
    NotFound(FragmentId),
    #[error("目录不存在: {0}")]
    DirectoryNotFound(PathBuf),
}

struct MemTable {
    entries: BTreeMap<FragmentId, HologramFragment>,
    source_groups: BTreeMap<u64, Vec<FragmentId>>,
    capacity: usize,
}

impl MemTable {
    fn new(capacity: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            source_groups: BTreeMap::new(),
            capacity,
        }
    }

    fn insert(&mut self, fragment: HologramFragment) {
        let id = fragment.id;
        let source_hash = fragment.metadata.source_hash;
        self.entries.insert(id, fragment);
        self.source_groups.entry(source_hash).or_default().push(id);
    }

    fn get(&self, id: FragmentId) -> Option<&HologramFragment> {
        self.entries.get(&id)
    }

    fn remove(&mut self, id: FragmentId) -> Option<HologramFragment> {
        let fragment = self.entries.remove(&id)?;
        let source_hash = fragment.metadata.source_hash;
        if let Some(ids) = self.source_groups.get_mut(&source_hash) {
            ids.retain(|&x| x != id);
            if ids.is_empty() {
                self.source_groups.remove(&source_hash);
            }
        }
        Some(fragment)
    }

    fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        self.entries.clear();
        self.source_groups.clear();
    }

    fn drain_all(&mut self) -> BTreeMap<FragmentId, HologramFragment> {
        self.source_groups.clear();
        std::mem::take(&mut self.entries)
    }

    fn get_by_source(&self, source_hash: u64) -> Vec<&HologramFragment> {
        self.source_groups
            .get(&source_hash)
            .map(|ids| ids.iter().filter_map(|id| self.entries.get(id)).collect())
            .unwrap_or_default()
    }

    fn all_fragments(&self) -> Vec<&HologramFragment> {
        self.entries.values().collect()
    }

    fn all_source_hashes(&self) -> Vec<u64> {
        self.source_groups.keys().copied().collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SsTableId {
    level: u32,
    seq: u32,
}

struct SsTable {
    id: SsTableId,
    path: PathBuf,
    entry_count: usize,
    min_key: FragmentId,
    max_key: FragmentId,
    source_index: BTreeMap<u64, Vec<FragmentId>>,
}

impl SsTable {
    fn path_for(dir: &Path, id: SsTableId) -> PathBuf {
        dir.join(format!("L{}-{}{}", id.level, id.seq, SSTABLE_EXT))
    }

    fn write_to_disk(
        dir: &Path,
        id: SsTableId,
        entries: &BTreeMap<FragmentId, HologramFragment>,
    ) -> Result<Self, LsmError> {
        let path = Self::path_for(dir, id);
        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);

        let entry_count = entries.len() as u64;
        let encoded_count = bincode::serialize(&entry_count)
            .map_err(|e| LsmError::Serialization(e.to_string()))?;
        writer.write_all(&encoded_count)?;

        let mut source_index: BTreeMap<u64, Vec<FragmentId>> = BTreeMap::new();
        let mut min_key = FragmentId::MAX;
        let mut max_key = FragmentId::MIN;

        for (&key, fragment) in entries {
            let encoded = bincode::serialize(fragment)
                .map_err(|e| LsmError::Serialization(e.to_string()))?;
            let len = encoded.len() as u64;
            let len_bytes = bincode::serialize(&len)
                .map_err(|e| LsmError::Serialization(e.to_string()))?;
            writer.write_all(&len_bytes)?;
            writer.write_all(&encoded)?;

            if key < min_key { min_key = key; }
            if key > max_key { max_key = key; }
            source_index
                .entry(fragment.metadata.source_hash)
                .or_default()
                .push(key);
        }
        writer.flush()?;

        Ok(Self {
            id,
            path,
            entry_count: entries.len(),
            min_key,
            max_key,
            source_index,
        })
    }

    fn read_entry(&self, id: FragmentId) -> Result<Option<HologramFragment>, LsmError> {
        if id < self.min_key || id > self.max_key {
            return Ok(None);
        }
        let file = fs::File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        let mut count_buf = vec![0u8; 8];
        reader.read_exact(&mut count_buf)?;
        let _count: u64 = bincode::deserialize(&count_buf)
            .map_err(|e| LsmError::Serialization(e.to_string()))?;

        loop {
            let mut len_buf = vec![0u8; 8];
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(_) => return Ok(None),
            }
            let entry_len: u64 = bincode::deserialize(&len_buf)
                .map_err(|e| LsmError::Serialization(e.to_string()))?;
            let mut entry_buf = vec![0u8; entry_len as usize];
            reader.read_exact(&mut entry_buf)?;
            let fragment: HologramFragment = bincode::deserialize(&entry_buf)
                .map_err(|e| LsmError::Serialization(e.to_string()))?;

            if fragment.id == id {
                return Ok(Some(fragment));
            }
            if fragment.id > id {
                return Ok(None);
            }
        }
    }

    fn read_all(&self) -> Result<BTreeMap<FragmentId, HologramFragment>, LsmError> {
        let file = fs::File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        let mut count_buf = vec![0u8; 8];
        reader.read_exact(&mut count_buf)?;
        let count: u64 = bincode::deserialize(&count_buf)
            .map_err(|e| LsmError::Serialization(e.to_string()))?;

        let mut entries = BTreeMap::new();
        for _ in 0..count {
            let mut len_buf = vec![0u8; 8];
            reader.read_exact(&mut len_buf)?;
            let entry_len: u64 = bincode::deserialize(&len_buf)
                .map_err(|e| LsmError::Serialization(e.to_string()))?;
            let mut entry_buf = vec![0u8; entry_len as usize];
            reader.read_exact(&mut entry_buf)?;
            let fragment: HologramFragment = bincode::deserialize(&entry_buf)
                .map_err(|e| LsmError::Serialization(e.to_string()))?;
            entries.insert(fragment.id, fragment);
        }
        Ok(entries)
    }

    fn delete_file(&self) -> Result<(), LsmError> {
        if self.path.exists() {
            fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    next_seq: u32,
    levels: Vec<Vec<(u32, u32)>>,
}

impl Manifest {
    fn new() -> Self {
        Self {
            next_seq: 0,
            levels: vec![vec![]],
        }
    }

    fn allocate_seq(&mut self) -> u32 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }

    fn ensure_level(&mut self, level: u32) {
        while self.levels.len() <= level as usize {
            self.levels.push(vec![]);
        }
    }

    fn add_sstable(&mut self, level: u32, seq: u32) {
        self.ensure_level(level);
        self.levels[level as usize].push((level, seq));
    }

    fn remove_sstable(&mut self, level: u32, seq: u32) {
        if let Some(lvl) = self.levels.get_mut(level as usize) {
            lvl.retain(|&(l, s)| !(l == level && s == seq));
        }
    }

    fn save(&self, dir: &Path) -> Result<(), LsmError> {
        let path = dir.join(MANIFEST_FILE);
        let encoded = bincode::serialize(self)
            .map_err(|e| LsmError::Serialization(e.to_string()))?;
        fs::write(path, encoded)?;
        Ok(())
    }

    fn load(dir: &Path) -> Result<Self, LsmError> {
        let path = dir.join(MANIFEST_FILE);
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = fs::read(path)?;
        bincode::deserialize(&data)
            .map_err(|e| LsmError::Serialization(e.to_string()))
    }
}

use serde::{Deserialize, Serialize};

pub struct LsmIndex {
    dir: PathBuf,
    memtable: MemTable,
    sstables: Vec<Vec<SsTable>>,
    manifest: Manifest,
    next_fragment_id: FragmentId,
    level0_limit: usize,
}

impl LsmIndex {
    pub fn open(dir: &Path) -> Result<Self, LsmError> {
        Self::open_with_capacity(dir, DEFAULT_MEMTABLE_CAPACITY)
    }

    pub fn open_with_capacity(dir: &Path, memtable_capacity: usize) -> Result<Self, LsmError> {
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }

        let manifest = Manifest::load(dir)?;
        let mut sstables: Vec<Vec<SsTable>> = vec![];

        for level_entries in &manifest.levels {
            let mut level_ssts = vec![];
            for &(level, seq) in level_entries {
                let id = SsTableId { level, seq };
                let path = SsTable::path_for(dir, id);
                if !path.exists() {
                    continue;
                }
                let all_entries = {
                    let file = fs::File::open(&path)?;
                    let mut reader = BufReader::new(file);
                    let mut count_buf = vec![0u8; 8];
                    reader.read_exact(&mut count_buf)?;
                    let count: u64 = bincode::deserialize(&count_buf)
                        .map_err(|e| LsmError::Serialization(e.to_string()))?;

                    let mut entries = BTreeMap::new();
                    let mut source_index: BTreeMap<u64, Vec<FragmentId>> = BTreeMap::new();
                    let mut min_key = FragmentId::MAX;
                    let mut max_key = FragmentId::MIN;

                    for _ in 0..count {
                        let mut len_buf = vec![0u8; 8];
                        reader.read_exact(&mut len_buf)?;
                        let entry_len: u64 = bincode::deserialize(&len_buf)
                            .map_err(|e| LsmError::Serialization(e.to_string()))?;
                        let mut entry_buf = vec![0u8; entry_len as usize];
                        reader.read_exact(&mut entry_buf)?;
                        let fragment: HologramFragment = bincode::deserialize(&entry_buf)
                            .map_err(|e| LsmError::Serialization(e.to_string()))?;

                        let key = fragment.id;
                        if key < min_key { min_key = key; }
                        if key > max_key { max_key = key; }
                        source_index
                            .entry(fragment.metadata.source_hash)
                            .or_default()
                            .push(key);
                        entries.insert(key, fragment);
                    }

                    (entries.len(), min_key, max_key, source_index)
                };

                let (entry_count, min_key, max_key, source_index) = all_entries;

                level_ssts.push(SsTable {
                    id,
                    path,
                    entry_count,
                    min_key,
                    max_key,
                    source_index,
                });
            }
            sstables.push(level_ssts);
        }

        let mut next_fragment_id: FragmentId = 1;
        for level in &sstables {
            for sst in level {
                if sst.max_key >= next_fragment_id {
                    next_fragment_id = sst.max_key + 1;
                }
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            memtable: MemTable::new(memtable_capacity),
            sstables,
            manifest,
            next_fragment_id,
            level0_limit: DEFAULT_LEVEL0_SSTABLE_LIMIT,
        })
    }

    pub fn insert(&mut self, fragment: HologramFragment) -> Result<FragmentId, LsmError> {
        let id = fragment.id;
        if id >= self.next_fragment_id {
            self.next_fragment_id = id + 1;
        }
        self.memtable.insert(fragment);

        if self.memtable.is_full() {
            self.flush_memtable()?;
        }
        Ok(id)
    }

    pub fn insert_batch(&mut self, fragments: Vec<HologramFragment>) -> Result<Vec<FragmentId>, LsmError> {
        let mut ids = Vec::with_capacity(fragments.len());
        for fragment in fragments {
            ids.push(self.insert(fragment)?);
        }
        Ok(ids)
    }

    pub fn get(&self, id: FragmentId) -> Result<Option<HologramFragment>, LsmError> {
        if let Some(fragment) = self.memtable.get(id) {
            return Ok(Some(fragment.clone()));
        }

        for level in &self.sstables {
            for sst in level.iter().rev() {
                if let Some(fragment) = sst.read_entry(id)? {
                    return Ok(Some(fragment));
                }
            }
        }

        Ok(None)
    }

    pub fn get_by_source(&self, source_hash: u64) -> Result<Vec<HologramFragment>, LsmError> {
        let mut results: BTreeMap<FragmentId, HologramFragment> = BTreeMap::new();

        for fragment in self.memtable.get_by_source(source_hash) {
            results.insert(fragment.id, fragment.clone());
        }

        for level in &self.sstables {
            for sst in level {
                if let Some(ids) = sst.source_index.get(&source_hash) {
                    for &id in ids {
                        if let std::collections::btree_map::Entry::Vacant(e) = results.entry(id) {
                            if let Some(fragment) = sst.read_entry(id)? {
                                e.insert(fragment);
                            }
                        }
                    }
                }
            }
        }

        Ok(results.into_values().collect())
    }

    pub fn remove(&mut self, id: FragmentId) -> Result<Option<HologramFragment>, LsmError> {
        if let Some(fragment) = self.memtable.remove(id) {
            return Ok(Some(fragment));
        }

        for level in &self.sstables {
            for sst in level.iter().rev() {
                if let Some(fragment) = sst.read_entry(id)? {
                    self.compact_and_remove(id)?;
                    return Ok(Some(fragment));
                }
            }
        }

        Ok(None)
    }

    pub fn integrity_check(&self, source_hash: u64) -> Result<IntegrityReport, LsmError> {
        let fragments = self.get_by_source(source_hash)?;
        let total = if let Some(first) = fragments.first() {
            first.metadata.fragment_count
        } else {
            0
        };
        Ok(IntegrityReport::new(total, fragments.len() as u32))
    }

    pub fn len(&self) -> usize {
        let mem_len = self.memtable.len();
        let sst_len: usize = self.sstables.iter().map(|l| l.iter().map(|s| s.entry_count).sum::<usize>()).sum();
        mem_len + sst_len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn all_fragments(&self) -> Result<Vec<HologramFragment>, LsmError> {
        let mut results: BTreeMap<FragmentId, HologramFragment> = BTreeMap::new();

        for fragment in self.memtable.all_fragments() {
            results.insert(fragment.id, fragment.clone());
        }

        for level in &self.sstables {
            for sst in level {
                let entries = sst.read_all()?;
                for (id, fragment) in entries {
                    results.entry(id).or_insert(fragment);
                }
            }
        }

        Ok(results.into_values().collect())
    }

    pub fn all_source_hashes(&self) -> Vec<u64> {
        let mut hashes: Vec<u64> = self.memtable.all_source_hashes();
        for level in &self.sstables {
            for sst in level {
                for &hash in sst.source_index.keys() {
                    if !hashes.contains(&hash) {
                        hashes.push(hash);
                    }
                }
            }
        }
        hashes
    }

    pub fn flush(&mut self) -> Result<(), LsmError> {
        if !self.memtable.is_empty() {
            self.flush_memtable()?;
        }
        Ok(())
    }

    pub fn compact(&mut self) -> Result<(), LsmError> {
        self.flush()?;

        while self.sstables.len() > 1 && self.level0_count() > self.level0_limit {
            self.compact_level0()?;
        }

        Ok(())
    }

    pub fn stats(&self) -> LsmStats {
        let mut level_stats = Vec::new();
        for (i, level) in self.sstables.iter().enumerate() {
            let sstable_count = level.len();
            let entry_count: usize = level.iter().map(|s| s.entry_count).sum();
            level_stats.push(LevelStats {
                level: i as u32,
                sstable_count,
                entry_count,
            });
        }
        LsmStats {
            memtable_entries: self.memtable.len(),
            level_stats,
            total_entries: self.len(),
        }
    }

    fn flush_memtable(&mut self) -> Result<(), LsmError> {
        if self.memtable.is_empty() {
            return Ok(());
        }

        let entries = self.memtable.drain_all();
        let seq = self.manifest.allocate_seq();
        let id = SsTableId { level: 0, seq };

        let sst = SsTable::write_to_disk(&self.dir, id, &entries)?;

        self.ensure_level(0);
        self.sstables[0].push(sst);
        self.manifest.add_sstable(0, seq);
        self.manifest.save(&self.dir)?;

        if self.level0_count() > self.level0_limit {
            self.compact_level0()?;
        }

        Ok(())
    }

    fn level0_count(&self) -> usize {
        self.sstables.first().map(|l| l.len()).unwrap_or(0)
    }

    fn ensure_level(&mut self, level: usize) {
        while self.sstables.len() <= level {
            self.sstables.push(vec![]);
        }
    }

    fn compact_level0(&mut self) -> Result<(), LsmError> {
        if self.sstables.is_empty() || self.sstables[0].len() <= 1 {
            return Ok(());
        }

        let mut merged: BTreeMap<FragmentId, HologramFragment> = BTreeMap::new();
        let mut old_ssts = Vec::new();
        let mut old_seqs = Vec::new();

        for sst in self.sstables[0].drain(..) {
            let entries = sst.read_all()?;
            for (id, fragment) in entries {
                merged.insert(id, fragment);
            }
            old_seqs.push(sst.id.seq);
            old_ssts.push(sst);
        }

        if merged.is_empty() {
            for sst in &old_ssts {
                sst.delete_file()?;
            }
            for seq in &old_seqs {
                self.manifest.remove_sstable(0, *seq);
            }
            self.manifest.save(&self.dir)?;
            return Ok(());
        }

        let seq = self.manifest.allocate_seq();
        let id = SsTableId { level: 1, seq };
        let new_sst = SsTable::write_to_disk(&self.dir, id, &merged)?;

        for sst in &old_ssts {
            sst.delete_file()?;
        }
        for seq in &old_seqs {
            self.manifest.remove_sstable(0, *seq);
        }
        self.manifest.add_sstable(1, id.seq);

        self.ensure_level(1);
        self.sstables[1].push(new_sst);

        self.manifest.save(&self.dir)?;
        Ok(())
    }

    fn compact_and_remove(&mut self, id: FragmentId) -> Result<(), LsmError> {
        let mut found = false;
        let mut all_entries: BTreeMap<FragmentId, HologramFragment> = BTreeMap::new();

        for level in &self.sstables {
            for sst in level {
                let entries = sst.read_all()?;
                for (k, v) in entries {
                    all_entries.entry(k).or_insert(v);
                }
            }
        }

        if all_entries.remove(&id).is_some() {
            found = true;
        }

        if !found {
            return Ok(());
        }

        let mut old_ssts = Vec::new();
        for level in &mut self.sstables {
            for sst in level.drain(..) {
                old_ssts.push(sst);
            }
        }
        for sst in &old_ssts {
            self.manifest.remove_sstable(sst.id.level, sst.id.seq);
            sst.delete_file()?;
        }

        if !all_entries.is_empty() {
            let seq = self.manifest.allocate_seq();
            let sst_id = SsTableId { level: 0, seq };
            let new_sst = SsTable::write_to_disk(&self.dir, sst_id, &all_entries)?;
            self.ensure_level(0);
            self.sstables[0].push(new_sst);
            self.manifest.add_sstable(0, seq);
        }

        self.manifest.save(&self.dir)?;
        Ok(())
    }
}

impl Drop for LsmIndex {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[derive(Debug)]
pub struct LevelStats {
    pub level: u32,
    pub sstable_count: usize,
    pub entry_count: usize,
}

#[derive(Debug)]
pub struct LsmStats {
    pub memtable_entries: usize,
    pub level_stats: Vec<LevelStats>,
    pub total_entries: usize,
}

impl std::fmt::Display for LsmStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LSM索引统计:")?;
        writeln!(f, "  MemTable: {} 条目", self.memtable_entries)?;
        for ls in &self.level_stats {
            writeln!(f, "  Level {}: {} SSTable, {} 条目", ls.level, ls.sstable_count, ls.entry_count)?;
        }
        write!(f, "  总计: {} 条目", self.total_entries)
    }
}
