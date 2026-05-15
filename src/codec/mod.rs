//! 编解码层：傅里叶编码器、全息分片器、冗余交织、稀疏编码、自适应窗口、RS纠删码、量子编码

pub mod fourier_encoder;
pub mod hologram_fragmenter;
pub mod redundancy_weaver;
pub mod parallel_encoder;
pub mod sparse_encoder;
pub mod adaptive_window;
pub mod reed_solomon;
pub mod quantum_encoder;
