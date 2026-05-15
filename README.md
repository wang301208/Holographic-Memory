# Holographic-Memory

全息记忆存储（Holographic Memory）— 基于 FFT 的高容错认知基础设施

## 概述

本项目实现了一套完整的全息记忆认知基础设施，基于卡尔·普里布拉姆的全息脑模型和丹尼斯·加博尔的全息术理论，将知识通过傅里叶变换映射到频域，使得每个存储片段都包含整体信息的缩影，从而实现极强的容错性和联想检索能力。v0.4.0 起升级为**认知基础设施**，新增全息推理、跨模态联想、自适应冗余三大引擎。

## 核心特性

- **频域全息编码**：使用 FFT + Hann 窗将知识映射到频域，完整解码精度 MSE ~1e-31
- **抗损毁性**：冗余交织编码 + 奇偶校验片段，支持 50% 数据损毁后恢复
- **Reed-Solomon 纠删码**：GF(2^8) 精确纠删重建，2 片校验可容 2 片丢失
- **联想检索**：频域余弦相似度 + 多跳联想图，相似概念自动聚类
- **全息推理引擎**：频域多头注意力 + 模式匹配推理 + 频域传播推理 + 推理链可解释性
- **跨模态联想**：文本↔图像频域桥接 + ModalityEncoder trait + 双向映射
- **自适应冗余**：四维重要性评分（访问频率/时效性/连接度/重建价值）→ 动态 RS 冗余决策
- **高存储效率**：频域压缩比优于传统向量数据库
- **并行编码**：rayon 数据并行，多核加速
- **WAL 持久化**：增量写入 + 紧凑化，支持崩溃恢复
- **零拷贝 mmap**：memmap2 直接映射，零拷贝读取 + 魔数校验
- **分层索引**：L0 热内存 + L1 LSM 磁盘，LRU 淘汰 + 自动升降级
- **量子启发编码**：叠加态 + 干涉模式 + Grover 放大 + 相位相干度
- **SIMD 加速**：4x 展开点积/加/减/缩放 + Hadamard 变换 + Walsh 编码
- **HTTP API**：axum 7 端点 + CORS（feature-gated）

## 对比传统向量数据库

| 维度 | ChromaDB / Qdrant | 全息存储 |
|------|-------------------|----------|
| 容错性 | 低（丢失即丢失） | 高（冗余编码 + 校验片段 + RS纠删 + 自适应冗余） |
| 联想能力 | 依赖相似度计算 | 天然支持全息联想 + 跨模态联想 |
| 推理能力 | 无 | 频域注意力推理 + 模式匹配 + 频域传播 |
| 存储效率 | 1x | 0.3x（压缩比更高） |
| 50%损毁恢复 | 不可能 | 可恢复 |
| 纠删编码 | 无 | GF(2^8) Reed-Solomon + 自适应冗余度 |
| 磁盘索引 | 单层 | 分层 L0/L1/L2 |

## 快速开始

### 构建与测试

```bash
cargo build
cargo test        # 150+ 个测试
cargo run --example basic_store_retrieve
cargo run --example fault_tolerance_demo
cargo run --example rs_fault_tolerance_demo
cargo run --example comprehensive_demo
cargo run --example cognitive_demo    # 认知基础设施演示
```

### CLI 工具

```bash
# 存储文件
holographic-memory store input.txt --data-dir ./data --redundancy 3

# 检索文件
holographic-memory retrieve --data-dir ./data

# 搜索相似内容
holographic-memory search "查询文本"

# 查看状态
holographic-memory status

# 容错性演示
holographic-memory demo
```

### HTTP API 服务

```bash
# 构建 HTTP 服务（需启用 http feature）
cargo build --features http

# 启动 API 服务
cargo run --bin holo-api --features http -- --addr 0.0.0.0:8080
```

端点：`/` `/status` `/store` `/retrieve` `/search` `/integrity` `/recover`

### 库使用

```rust
use holographic_memory::*;

// 基础使用
let config = HolographicConfig::default();
let mut hm = HolographicMemory::new(config);

// 存储
let data: Vec<f64> = vec![/* 你的数据 */];
let result = hm.store(&data).unwrap();

// 检索
let decoded = hm.retrieve(result.source_hash, data.len()).unwrap();

// 搜索
let results = hm.search(&query, 10).unwrap();
```

### 认知基础设施（v0.4.0+）

```rust
use holographic_memory::*;

let config = HolographicConfig::default();
let mut hm = HolographicMemory::new(config)
    .with_reasoner(AttentionConfig::default())
    .with_cross_modal()
    .with_adaptive_redundancy(RedundancyStrategy::default());

// 自适应存储（含重要性评估）
let result = hm.adaptive_store(&data).unwrap();
println!("重要性: {:?}", result.redundancy_decision.importance.level);
println!("RS校验片: {}", result.redundancy_decision.rs_parity_shards);

// 全息推理
let inference = hm.reason(&query, 5).unwrap();
println!("推理相干度: {:.4}", inference.coherence_score);

// 跨模态联想检索
let cross = hm.cross_modal_search(&text_data, &Modality::Text, &Modality::Image, 5).unwrap();
for assoc in &cross {
    println!("桥接置信度: {:.4}", assoc.bridge_confidence);
}
```

### 高级配置

```rust
use holographic_memory::*;
use std::path::PathBuf;

let config = HolographicConfig::default();

// 分层索引 + RS纠删码 + mmap持久化 + 认知引擎
let mut hm = HolographicMemory::new(config)
    .with_tiered_index(TieredConfig {
        l0_capacity: 10000,
        l1_memtable_capacity: 1000,
        l1_dir: PathBuf::from("./data/l1"),
        promote_threshold: 5,
        demote_after_access: false,
    })
    .unwrap()
    .with_reed_solomon(4, 2)
    .unwrap()
    .with_mmap("./data/mmap")
    .with_reasoner(AttentionConfig::default())
    .with_cross_modal()
    .with_adaptive_redundancy(RedundancyStrategy::default());

// RS 纠删码保护存储
let result = hm.store_with_rs(&data).unwrap();

// mmap 零拷贝持久化
hm.save_mmap("snapshot.mmap").unwrap();
```

## 项目结构

```
src/
├── lib.rs                       # 库入口（40+ 公开 API）
├── holographic.rs               # 统一高级 API（HolographicMemory）
├── types.rs                     # 核心类型定义
├── bin/
│   ├── holographic-memory.rs    # CLI 工具
│   └── holo-api.rs              # HTTP API 服务（feature-gated）
├── foundation/                  # 基础层
│   ├── math.rs                  # FFT / 2D-FFT / 相似度 / 范数
│   ├── memory_pool.rs           # 64 字节对齐内存池
│   ├── config.rs                # TOML 配置 + 参数校验
│   └── simd_ops.rs              # SIMD 4x展开 + Hadamard + Walsh
├── codec/                       # 编解码层
│   ├── fourier_encoder.rs       # Hann 窗 + FFT 编码 / OLA 解码
│   ├── hologram_fragmenter.rs   # 全息分片 / 反分片
│   ├── redundancy_weaver.rs     # 奇偶校验 + 冗余交织 / 恢复
│   ├── parallel_encoder.rs      # rayon 并行编码
│   ├── sparse_encoder.rs        # 稀疏频域编码 + 能量分析
│   ├── adaptive_window.rs       # 自适应窗口选择器
│   ├── reed_solomon.rs          # GF(2^8) Reed-Solomon 纠删码
│   ├── quantum_encoder.rs       # 量子启发编码
│   └── cross_modal.rs           # 跨模态联想引擎
├── storage/                     # 存储层
│   ├── holographic_index.rs     # 全息索引 + 完整性检查
│   ├── segment_manager.rs       # 段落管理
│   ├── persistence.rs           # WAL 增量持久化 + 紧凑化
│   ├── lsm_index.rs             # LSM 磁盘索引 + Compaction
│   ├── mmap_persistence.rs      # 零拷贝 mmap 持久化
│   ├── tiered_index.rs          # 分层索引 L0/L1/L2
│   └── adaptive_redundancy.rs   # 自适应冗余引擎
├── retrieval/                   # 检索层
│   ├── similarity_matcher.rs    # 频域余弦相似度 + Top-K
│   ├── associative_search.rs    # 多跳联想检索引擎
│   ├── partial_recovery.rs      # 部分损坏恢复 + 置信度评估
│   └── holographic_reasoner.rs  # 全息推理引擎
└── api/
    └── http.rs                  # axum HTTP API（feature-gated）
```

## 技术栈

- **语言**：Rust（内存安全 + 零成本抽象 + 高性能）
- **FFT**：rustfft（纯 Rust FFT 实现）
- **线性代数**：ndarray（N 维数组）
- **并行**：rayon（数据并行）
- **序列化**：serde + bincode（高效二进制）
- **配置**：TOML
- **mmap**：memmap2（零拷贝文件映射）
- **HTTP**：axum + tokio + tower-http（feature-gated）

## 测试覆盖

- **150+ 个测试**，覆盖：
  - 核心类型、FFT 往返、编解码精度、全息分片、冗余交织
  - 奇偶校验恢复、容错性扫描、索引管理、持久化往返
  - WAL 增量写入、联想检索、相似度匹配、端到端存取
  - 并行编码、稀疏编码、自适应窗口、统一高级 API
  - LSM 磁盘索引、mmap 零拷贝持久化、Reed-Solomon 纠删码
  - 量子启发编码、SIMD 加速、分层索引
  - 后端集成（TieredIndex+RS+Mmap+HolographicMemory 端到端）
  - 认知基础设施（全息推理 + 跨模态联想 + 自适应冗余 + 统一 API）

## 版本历史

- **v0.4.0**：全息推理引擎、跨模态联想引擎、自适应冗余引擎、统一认知 API
- **v0.3.0**：量子启发编码、SIMD 加速、分层索引、RS 纠删码、mmap 持久化、HTTP API
- **v0.2.0**：稀疏频域编码、自适应窗口、LSM 磁盘索引、Reed-Solomon 纠删码
- **v0.1.0**：核心 FFT 编解码、冗余交织、全息索引、联想检索、WAL 持久化、CLI

## 许可证

MIT
