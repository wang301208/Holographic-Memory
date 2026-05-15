use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::fs;

use crate::storage::holographic_index::HolographicIndex;
use crate::types::HologramFragment;

#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("序列化错误: {0}")]
    Serialize(#[from] bincode::Error),
}

pub struct PersistenceEngine {
    data_dir: PathBuf,
    wal: Vec<WalEntry>,
    wal_path: PathBuf,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct WalEntry {
    action: WalAction,
    fragment: HologramFragment,
}

#[derive(serde::Serialize, serde::Deserialize)]
enum WalAction {
    Insert,
    Delete,
}

impl PersistenceEngine {
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        let data_dir = data_dir.as_ref().to_path_buf();
        let wal_path = data_dir.join("wal.bin");
        Self {
            data_dir,
            wal: Vec::new(),
            wal_path,
        }
    }

    pub fn save_index(&self, index: &HolographicIndex, filename: &str) -> Result<(), PersistenceError> {
        fs::create_dir_all(&self.data_dir)?;
        let path = self.data_dir.join(filename);
        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);

        let fragments: Vec<&HologramFragment> = index.all_fragments();
        let data = bincode::serialize(&fragments)?;
        let len = data.len() as u64;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&data)?;
        writer.flush()?;
        Ok(())
    }

    pub fn load_index(&self, filename: &str) -> Result<HolographicIndex, PersistenceError> {
        let path = self.data_dir.join(filename);
        let file = fs::File::open(&path)?;
        let mut reader = BufReader::new(file);

        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;

        let fragments: Vec<HologramFragment> = bincode::deserialize(&data)?;
        let mut index = HolographicIndex::new();
        for fragment in fragments {
            index.insert(fragment);
        }
        Ok(index)
    }

    pub fn save_index_roundtrip(&self, index: &HolographicIndex, filename: &str) -> Result<HolographicIndex, PersistenceError> {
        self.save_index(index, filename)?;
        self.load_index(filename)
    }

    pub fn insert_incremental(&mut self, fragment: &HologramFragment) -> Result<(), PersistenceError> {
        self.wal.push(WalEntry {
            action: WalAction::Insert,
            fragment: fragment.clone(),
        });
        if self.wal.len() >= 100 {
            self.flush_wal()?;
        }
        Ok(())
    }

    pub fn flush_wal(&mut self) -> Result<(), PersistenceError> {
        if self.wal.is_empty() {
            return Ok(());
        }

        fs::create_dir_all(&self.data_dir)?;
        let exists = self.wal_path.exists();
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)?;
        let mut writer = BufWriter::new(file);

        if !exists {
            let header = b"HOLO_WAL_V1\0";
            writer.write_all(header)?;
        }

        for entry in self.wal.drain(..) {
            let data = bincode::serialize(&entry)?;
            let len = data.len() as u32;
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&data)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn replay_wal(&self) -> Result<Vec<HologramFragment>, PersistenceError> {
        if !self.wal_path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.wal_path)?;
        let mut reader = BufReader::new(file);

        let mut header = [0u8; 12];
        reader.read_exact(&mut header)?;
        if &header != b"HOLO_WAL_V1\0" {
            return Err(PersistenceError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "无效的WAL头部",
            )));
        }

        let mut fragments = Vec::new();
        loop {
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(PersistenceError::Io(e)),
            }
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;

            let entry: WalEntry = bincode::deserialize(&data)?;
            match entry.action {
                WalAction::Insert => fragments.push(entry.fragment),
                WalAction::Delete => {
                    let id = entry.fragment.id;
                    fragments.retain(|f| f.id != id);
                }
            }
        }

        Ok(fragments)
    }

    pub fn compact(&mut self, index: &HolographicIndex, filename: &str) -> Result<(), PersistenceError> {
        self.save_index(index, filename)?;
        if self.wal_path.exists() {
            fs::remove_file(&self.wal_path)?;
        }
        self.wal.clear();
        Ok(())
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}
