#![allow(unsafe_code)]

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::types::HologramFragment;
use crate::storage::holographic_index::HolographicIndex;

const MAGIC: &[u8; 8] = b"HOLOMM01";
const HEADER_LEN: usize = 8 + 8;

#[derive(Debug, thiserror::Error)]
pub enum MmapError {
    #[error("IO错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("序列化错误: {0}")]
    Serialization(String),
    #[error("无效文件格式: {0}")]
    InvalidFormat(String),
    #[error("文件过小: 期望至少{expected}字节, 实际{actual}字节")]
    FileTooSmall { expected: usize, actual: usize },
}

pub struct MmapPersistence {
    dir: PathBuf,
}

pub struct MmapReader {
    _mmap: Mmap,
    data_ptr: *const u8,
    data_len: usize,
}

impl MmapReader {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.data_len) }
    }

    pub fn data_len(&self) -> usize {
        self.data_len
    }

    pub fn load_fragments(&self) -> Result<Vec<HologramFragment>, MmapError> {
        let slice = self.as_slice();
        bincode::deserialize(slice)
            .map_err(|e| MmapError::Serialization(e.to_string()))
    }

    pub fn load_index(&self) -> Result<HolographicIndex, MmapError> {
        let fragments = self.load_fragments()?;
        let mut index = HolographicIndex::new();
        for fragment in fragments {
            index.insert(fragment);
        }
        Ok(index)
    }
}

unsafe impl Send for MmapReader {}
unsafe impl Sync for MmapReader {}

impl MmapPersistence {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }

    pub fn write(
        &self,
        fragments: &[HologramFragment],
        filename: &str,
    ) -> Result<PathBuf, MmapError> {
        fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(filename);

        let serialized = bincode::serialize(fragments)
            .map_err(|e| MmapError::Serialization(e.to_string()))?;

        let total_len = HEADER_LEN + serialized.len();
        let file = fs::File::create(&path)?;
        file.set_len(total_len as u64)?;

        let mut writer = BufWriter::new(file);
        writer.write_all(MAGIC)?;
        writer.write_all(&(serialized.len() as u64).to_le_bytes())?;
        writer.write_all(&serialized)?;
        writer.flush()?;

        Ok(path)
    }

    pub fn write_index(
        &self,
        index: &HolographicIndex,
        filename: &str,
    ) -> Result<PathBuf, MmapError> {
        let fragments: Vec<&HologramFragment> = index.all_fragments();
        let owned: Vec<HologramFragment> = fragments.into_iter().cloned().collect();
        self.write(&owned, filename)
    }

    pub fn read(&self, filename: &str) -> Result<MmapReader, MmapError> {
        let path = self.dir.join(filename);
        let file = fs::File::open(&path)?;
        let metadata = file.metadata()?;

        if metadata.len() < HEADER_LEN as u64 {
            return Err(MmapError::FileTooSmall {
                expected: HEADER_LEN,
                actual: metadata.len() as usize,
            });
        }

        let mmap = unsafe { Mmap::map(&file)? };

        if &mmap[..8] != MAGIC {
            return Err(MmapError::InvalidFormat(
                format!("魔数不匹配: 期望{:?}, 实际{:?}", MAGIC, &mmap[..8])
            ));
        }

        let data_len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let file_data_len = mmap.len() - HEADER_LEN;
        if data_len > file_data_len {
            return Err(MmapError::FileTooSmall {
                expected: HEADER_LEN + data_len,
                actual: mmap.len(),
            });
        }

        Ok(MmapReader {
            data_ptr: mmap[HEADER_LEN..].as_ptr(),
            data_len,
            _mmap: mmap,
        })
    }

    pub fn read_fragments(&self, filename: &str) -> Result<Vec<HologramFragment>, MmapError> {
        let reader = self.read(filename)?;
        reader.load_fragments()
    }

    pub fn read_index(&self, filename: &str) -> Result<HolographicIndex, MmapError> {
        let reader = self.read(filename)?;
        reader.load_index()
    }

    pub fn file_size(&self, filename: &str) -> Result<u64, MmapError> {
        let path = self.dir.join(filename);
        let metadata = fs::metadata(path)?;
        Ok(metadata.len())
    }

    pub fn exists(&self, filename: &str) -> bool {
        self.dir.join(filename).exists()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}
