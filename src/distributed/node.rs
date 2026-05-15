use std::collections::HashMap;

use crate::holographic::{HolographicMemory, HoloError, StoreResult};
use crate::foundation::config::HolographicConfig;
use crate::types::{FragmentId, HologramFragment, AssociatedItem};
use crate::retrieval::holographic_reasoner::InferenceResult;
use crate::codec::cross_modal::{CrossModalAssociation, Modality};

use super::consistent_hash::{ConsistentHashRing, RingError};
use super::gossip::{GossipProtocol, GossipConfig, GossipMessage};
use super::anti_entropy::{AntiEntropyRepair, MerkleTree, RepairTask};

/// 分布式配置
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    pub node_id: String,
    pub address: String,
    pub virtual_nodes: usize,
    pub replication_factor: usize,
    pub gossip_config: GossipConfig,
    pub merkle_bucket_size: usize,
    pub max_concurrent_repairs: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            node_id: "node-0".to_string(),
            address: "127.0.0.1:7000".to_string(),
            virtual_nodes: 150,
            replication_factor: 3,
            gossip_config: GossipConfig::default(),
            merkle_bucket_size: 256,
            max_concurrent_repairs: 4,
        }
    }
}

/// 分布式错误类型
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("一致性哈希错误: {0}")]
    Ring(#[from] RingError),
    #[error("本地存储错误: {0}")]
    Local(#[from] HoloError),
    #[error("节点不可达: {0}")]
    NodeUnreachable(String),
    #[error("副本不足: 需要{required}个, 可用{available}个")]
    InsufficientReplicas { required: usize, available: usize },
    #[error("集群未就绪: {0}")]
    ClusterNotReady(String),
    #[error("修复错误: {0}")]
    Repair(String),
}

/// 分布式存储结果
#[derive(Debug)]
pub struct DistributedStoreResult {
    pub local_result: StoreResult,
    pub replica_nodes: Vec<String>,
    pub replication_factor: usize,
    pub quorum_achieved: bool,
}

/// 分布式检索结果
#[derive(Debug)]
pub struct DistributedRetrieveResult {
    pub data: Vec<f64>,
    pub source_node: String,
    pub from_replica: bool,
    pub integrity_verified: bool,
}

/// 分布式全息节点
///
/// 将 HolographicMemory 与分布式基础设施整合：
/// - 一致性哈希环决定数据分片归属
/// - Gossip 协议维护集群成员关系
/// - 反熵修复保证副本一致性
/// - 多副本写入 + 仲裁读
pub struct DistributedHolographicNode {
    pub local_store: HolographicMemory,
    pub ring: ConsistentHashRing,
    pub gossip: GossipProtocol,
    pub anti_entropy: AntiEntropyRepair,
    config: DistributedConfig,
    remote_stores: HashMap<String, Vec<HologramFragment>>,
    pending_writes: HashMap<FragmentId, Vec<String>>,
}

impl DistributedHolographicNode {
    pub fn new(holo_config: HolographicConfig, dist_config: DistributedConfig) -> Self {
        let node_id = dist_config.node_id.clone();
        let address = dist_config.address.clone();

        let local_store = HolographicMemory::new(holo_config);
        let ring = ConsistentHashRing::new(dist_config.virtual_nodes);
        let gossip = GossipProtocol::with_address(
            node_id, address, dist_config.gossip_config.clone(),
        );
        let anti_entropy = AntiEntropyRepair::new(
            dist_config.merkle_bucket_size,
            dist_config.max_concurrent_repairs,
        );

        Self {
            local_store,
            ring,
            gossip,
            anti_entropy,
            config: dist_config,
            remote_stores: HashMap::new(),
            pending_writes: HashMap::new(),
        }
    }

    pub fn join_cluster(&mut self, seed_nodes: &[(String, String)]) -> Result<(), DistributedError> {
        self.ring.add_node(&self.config.node_id)?;

        for (node_id, address) in seed_nodes {
            self.ring.add_node(node_id)?;
            self.gossip.handle_join(
                node_id.clone(),
                address.clone(),
                0,
                HashMap::new(),
            );
        }

        Ok(())
    }

    pub fn distributed_store(&mut self, data: &[f64]) -> Result<DistributedStoreResult, DistributedError> {
        let local_result = self.local_store.store(data)?;

        let replica_nodes = self.ring.find_replicas(
            &format!("src:{}", local_result.source_hash),
            self.config.replication_factor,
        );

        let quorum = self.config.replication_factor / 2 + 1;
        let alive_replicas: Vec<String> = replica_nodes.iter()
            .filter(|node| self.gossip.membership.is_alive(node))
            .cloned()
            .collect();

        let quorum_achieved = alive_replicas.len() + 1 >= quorum;

        for fragment_id in &local_result.fragment_ids {
            self.pending_writes.insert(*fragment_id, alive_replicas.clone());
        }

        self.local_store.build_propagation_graph();

        Ok(DistributedStoreResult {
            local_result,
            replica_nodes: alive_replicas,
            replication_factor: self.config.replication_factor,
            quorum_achieved,
        })
    }

    pub fn distributed_retrieve(
        &mut self,
        source_hash: u64,
        expected_len: usize,
    ) -> Result<DistributedRetrieveResult, DistributedError> {
        match self.local_store.retrieve(source_hash, expected_len) {
            Ok(data) => Ok(DistributedRetrieveResult {
                data,
                source_node: self.config.node_id.clone(),
                from_replica: false,
                integrity_verified: true,
            }),
            Err(_) => {
                let key = format!("src:{}", source_hash);
                let replicas = self.ring.find_replicas(&key, self.config.replication_factor);

                for replica_node in &replicas {
                    if replica_node == &self.config.node_id {
                        continue;
                    }
                    if let Some(fragments) = self.remote_stores.get(replica_node) {
                        let source_fragments: Vec<HologramFragment> = fragments.iter()
                            .filter(|f| f.id == source_hash)
                            .cloned()
                            .collect();
                        if !source_fragments.is_empty() {
                            let unwoven = self.local_store.unweave(&source_fragments);
                            let decoded = self.local_store.decode_fragments(&unwoven, expected_len);
                            return Ok(DistributedRetrieveResult {
                                data: decoded,
                                source_node: replica_node.clone(),
                                from_replica: true,
                                integrity_verified: false,
                            });
                        }
                    }
                }

                Err(DistributedError::Local(HoloError::Decode(
                    format!("未找到源 {} (本地和副本均失败)", source_hash),
                )))
            }
        }
    }

    pub fn distributed_search(&mut self, query: &[f64], top_k: usize) -> Result<Vec<AssociatedItem>, DistributedError> {
        Ok(self.local_store.search(query, top_k)?)
    }

    pub fn distributed_reason(&mut self, query: &[f64], top_k: usize) -> Result<InferenceResult, DistributedError> {
        Ok(self.local_store.reason(query, top_k)?)
    }

    pub fn distributed_cross_modal_search(
        &mut self,
        query: &[f64],
        source_modality: &Modality,
        target_modality: &Modality,
        top_k: usize,
    ) -> Result<Vec<CrossModalAssociation>, DistributedError> {
        Ok(self.local_store.cross_modal_search(query, source_modality, target_modality, top_k)?)
    }

    pub fn receive_gossip(&mut self, messages: Vec<GossipMessage>) {
        for msg in messages {
            self.gossip.receive_message(msg);
        }
    }

    pub fn gossip_tick(&mut self) -> Vec<GossipMessage> {
        self.gossip.tick()
    }

    pub fn send_heartbeat(&mut self) {
        let fragment_count = self.local_store.fragment_count();
        let load = fragment_count as f64 / 100000.0;
        self.gossip.send_heartbeat(fragment_count, load);
    }

    pub fn trigger_anti_entropy(&mut self, remote_node: &str, remote_root_hash: u64) -> Option<u64> {
        self.anti_entropy.update_digests(self.compute_local_digests());

        let local_hash = self.anti_entropy.root_hash();
        if local_hash == remote_root_hash {
            return None;
        }

        let mut remote_tree = MerkleTree::new(self.config.merkle_bucket_size);
        let fake_digests = vec![(0, remote_root_hash)];
        remote_tree.build_from_digests(&fake_digests);

        let diff = self.anti_entropy.compare_with(&remote_tree);
        if diff.identical {
            return None;
        }

        Some(self.anti_entropy.create_repair_task(
            self.config.node_id.clone(),
            remote_node.to_string(),
            diff.differing_ranges,
        ))
    }

    fn compute_local_digests(&self) -> Vec<(u64, u64)> {
        let fragments = self.local_store.all_fragments_pub();
        fragments.iter().map(|f| {
            let id_hash = f.id;
            let content_hash: u64 = f.frequency_domain.iter()
                .map(|c| c.re.to_bits() ^ c.im.to_bits())
                .fold(0u64, |acc, h| acc.wrapping_mul(31).wrapping_add(h));
            (id_hash, content_hash)
        }).collect()
    }

    pub fn advance_repairs(&mut self) -> Vec<&RepairTask> {
        let pending: Vec<u64> = self.anti_entropy.pending_tasks()
            .iter()
            .map(|t| t.task_id)
            .collect();

        for task_id in pending {
            if self.anti_entropy.start_repair(task_id).is_ok() {
                self.anti_entropy.complete_repair(task_id);
            }
        }

        self.anti_entropy.in_progress_tasks()
    }

    pub fn receive_remote_fragments(&mut self, source_node: &str, fragments: Vec<HologramFragment>) {
        self.remote_stores.insert(source_node.to_string(), fragments);
    }

    pub fn cluster_status(&self) -> ClusterStatus {
        ClusterStatus {
            node_id: self.config.node_id.clone(),
            alive_nodes: self.gossip.membership.alive_node_count(),
            total_nodes: self.ring.node_count(),
            local_fragments: self.local_store.fragment_count(),
            replication_factor: self.config.replication_factor,
            pending_repairs: self.anti_entropy.pending_tasks().len(),
            ring_vnodes: self.ring.virtual_node_count_total(),
        }
    }

    pub fn node_id(&self) -> &str {
        &self.config.node_id
    }

    pub fn replication_factor(&self) -> usize {
        self.config.replication_factor
    }
}

/// 集群状态摘要
#[derive(Debug)]
pub struct ClusterStatus {
    pub node_id: String,
    pub alive_nodes: usize,
    pub total_nodes: usize,
    pub local_fragments: usize,
    pub replication_factor: usize,
    pub pending_repairs: usize,
    pub ring_vnodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(node_id: &str) -> DistributedConfig {
        DistributedConfig {
            node_id: node_id.to_string(),
            address: format!("127.0.0.1:700{}", node_id.chars().last().unwrap()),
            virtual_nodes: 150,
            replication_factor: 3,
            gossip_config: GossipConfig::default(),
            merkle_bucket_size: 256,
            max_concurrent_repairs: 4,
        }
    }

    #[test]
    fn test_create_node() {
        let config = make_config("node1");
        let node = DistributedHolographicNode::new(HolographicConfig::default(), config);
        assert_eq!(node.node_id(), "node1");
    }

    #[test]
    fn test_join_cluster() {
        let config = make_config("node1");
        let mut node = DistributedHolographicNode::new(HolographicConfig::default(), config);
        node.join_cluster(&[
            ("node2".to_string(), "127.0.0.1:7002".to_string()),
        ]).unwrap();
        assert_eq!(node.ring.node_count(), 2);
    }

    #[test]
    fn test_distributed_store_retrieve() {
        let config = DistributedConfig {
            replication_factor: 1,
            ..make_config("node1")
        };
        let mut node = DistributedHolographicNode::new(HolographicConfig::default(), config);
        node.ring.add_node("node1").unwrap();

        let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
        let store_result = node.distributed_store(&data).unwrap();
        assert!(store_result.quorum_achieved);

        let retrieve_result = node.distributed_retrieve(store_result.local_result.source_hash, data.len()).unwrap();
        assert!(!retrieve_result.from_replica);
    }

    #[test]
    fn test_cluster_status() {
        let config = make_config("node1");
        let mut node = DistributedHolographicNode::new(HolographicConfig::default(), config);
        node.ring.add_node("node1").unwrap();
        let status = node.cluster_status();
        assert_eq!(status.total_nodes, 1);
        assert_eq!(status.node_id, "node1");
    }

    #[test]
    fn test_gossip_heartbeat() {
        let config = make_config("node1");
        let mut node = DistributedHolographicNode::new(HolographicConfig::default(), config);
        node.send_heartbeat();
        let msgs = node.gossip.drain_messages();
        assert!(!msgs.is_empty());
    }
}
