#![allow(unsafe_code)]

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

pub struct MemoryPool {
    chunks: Vec<PoolChunk>,
    default_chunk_size: usize,
}

struct PoolChunk {
    ptr: NonNull<u8>,
    layout: Layout,
    used: usize,
    capacity: usize,
}

impl PoolChunk {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 64).expect("无效的内存布局");
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).expect("内存分配失败");
        PoolChunk {
            ptr,
            layout,
            used: 0,
            capacity: size,
        }
    }

    fn allocate(&mut self, size: usize) -> Option<NonNull<u8>> {
        let aligned_size = (size + 63) & !63;
        if self.used + aligned_size > self.capacity {
            return None;
        }
        let offset = self.ptr.as_ptr() as usize + self.used;
        self.used += aligned_size;
        NonNull::new(offset as *mut u8)
    }
}

impl Drop for PoolChunk {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

unsafe impl Send for PoolChunk {}

impl MemoryPool {
    pub fn new(default_chunk_size: usize) -> Self {
        MemoryPool {
            chunks: Vec::new(),
            default_chunk_size,
        }
    }

    pub fn allocate(&mut self, size: usize) -> NonNull<u8> {
        for chunk in &mut self.chunks {
            if let Some(ptr) = chunk.allocate(size) {
                return ptr;
            }
        }
        let chunk_size = size.max(self.default_chunk_size);
        let mut chunk = PoolChunk::new(chunk_size);
        let ptr = chunk.allocate(size).expect("新分配的块必须可用");
        self.chunks.push(chunk);
        ptr
    }

    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.used = 0;
        }
    }

    pub fn total_used(&self) -> usize {
        self.chunks.iter().map(|c| c.used).sum()
    }

    pub fn total_capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.capacity).sum()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new(1024 * 1024)
    }
}
