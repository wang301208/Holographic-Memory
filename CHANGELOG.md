# Changelog

本文件记录全息记忆存储引擎的所有版本变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/)。

## [0.3.0] - 2025-05-15

### 新增

- 零拷贝 mmap 持久化：`MmapPersistence`（memmap2 直接映射 + 魔数校验）
- 分层全息索引：`TieredIndex`（L0 热内存 + L1 LSM 磁盘，LRU 淘汰 + promote/demote + 穿透查询）
- 量子启发编码：`QuantumEncoder`（叠加态 + 干涉模式 + Grover 放大 + 相位相干度 + 熵测量）
- SIMD 加速运算：`SimdOps`（4x 展开点积/加/减/缩放 + Hadamard 变换 + Walsh 编码）
- axum HTTP API：7 端点（`/` `/status` `/store` `/retrieve` `/search` `/integrity` `/recover`）+ CORS（feature-gated `http`）
- HTTP 服务端独立二进制：`src/bin/holo-api.rs`（feature-gated `http`）
- `HolographicMemory` 增强：`with_tiered_index()` + `with_reed_solomon()` + `with_mmap()` + `store_with_rs()` + `save_mmap()/load_mmap()` + 双后端 `IndexBackend` 枚举
- 后端集成测试：11 个 TieredIndex+RS+Mmap+HolographicMemory 端到端测试
- RS 纠删码容错实战示例：`examples/rs_fault_tolerance_demo.rs`
- 综合示例：`examples/comprehensive_demo.rs`（7 大模块演示）
- criterion 基准测试：7 组性能基线（FFT 编码/解码、相似度检索、RS 编码/重建、SIMD、HM 存储、冗余交织）（feature-gated `bench`）
- 关键 API 文档注释（`HolographicMemory`、`HoloError`、`StoreResult`、`FaultToleranceResult` + builder 方法）

### 优化

- FFT plan 缓存：`FourierTransformer` 缓存 forward/inverse FFT plan，避免重复创建
- Hann 窗预计算：`FourierEncoder` 构造时预计算 Hann 窗表，编码/解码内循环零三角函数调用
- `IndexBackend` 大变体优化：`Tiered(TieredIndex)` → `Tiered(Box<TieredIndex>)`，减少枚举大小
- clippy 零警告

### 变更

- README.md 全面更新至 v0.3.0
- ROADMAP.md 新增版本迭代记录（v0.1.0~v0.5.0）
- CI 流水线新增 http feature 测试、基准测试编译检查、http feature clippy
- Cargo.toml 新增 `http`、`bench` feature，`holo-api` `[[bin]]`

## [0.2.0] - 2025-05-14

### 新增

- 稀疏频域编码：`SparseEncoder`（Top-K 系数保留 + 能量分析）
- 自适应 FFT 窗口：`AdaptiveWindowSelector`（谱平坦度/ZCR/RMS → 动态窗口/重叠率）
- LSM 磁盘索引：`LsmIndex`（MemTable + SSTable + Compaction + 重启恢复）
- GF(2^8) Reed-Solomon 纠删码：`ReedSolomon`（Vandermonde 矩阵编码/精确纠删重建/验证）

## [0.1.0] - 2025-05-13

### 新增

- 基础搭建：Cargo 项目 + 核心类型(6种) + FFT 数学原语 + 内存池 + TOML 配置系统
- 编码引擎：傅里叶编解码器(Hann窗+OLA) + 全息分片器 + 冗余交织器(奇偶校验+配对冗余) + 并行编码(rayon)
- 存储引擎：全息索引(HashMap) + 段落管理 + WAL 增量持久化 + 紧凑化
- 检索引擎：频域余弦相似度 + Top-K + 多跳联想检索引擎 + 部分损坏恢复
- 集成验证：端到端存取 + 30% 容错验证 + 持久化往返
- API/SDK：CLI 工具(6命令) + HTTP API 骨架 + 4 个示例
- 配置/CI：.cargo/config.toml + GitHub Actions CI
- 统一高级 API：`HolographicMemory` 结构体 + `HoloError` 统一错误 + 线程安全修复
