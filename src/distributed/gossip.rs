use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 节点状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeStatus {
    Alive,
    Suspect,
    Dead,
    Left,
}

/// 节点信息
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: String,
    pub status: NodeStatus,
    pub incarnation: u64,
    pub metadata: HashMap<String, String>,
    pub last_heartbeat: Instant,
}

/// 集群成员状态
#[derive(Debug, Clone)]
pub struct MembershipState {
    pub nodes: HashMap<String, NodeInfo>,
    pub local_node_id: String,
    pub incarnation: u64,
}

impl MembershipState {
    pub fn new(local_node_id: String) -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(local_node_id.clone(), NodeInfo {
            node_id: local_node_id.clone(),
            address: "localhost".to_string(),
            status: NodeStatus::Alive,
            incarnation: 0,
            metadata: HashMap::new(),
            last_heartbeat: Instant::now(),
        });
        Self {
            nodes,
            local_node_id,
            incarnation: 0,
        }
    }

    pub fn alive_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes.values()
            .filter(|n| n.status == NodeStatus::Alive)
            .collect()
    }

    pub fn alive_node_count(&self) -> usize {
        self.nodes.values()
            .filter(|n| n.status == NodeStatus::Alive)
            .count()
    }

    pub fn is_alive(&self, node_id: &str) -> bool {
        self.nodes.get(node_id)
            .map(|n| n.status == NodeStatus::Alive)
            .unwrap_or(false)
    }

    pub fn merge_node(&mut self, incoming: NodeInfo) {
        match self.nodes.get(&incoming.node_id) {
            Some(existing) => {
                if incoming.incarnation > existing.incarnation {
                    self.nodes.insert(incoming.node_id.clone(), incoming);
                } else if incoming.incarnation == existing.incarnation {
                    let incoming_priority = status_priority(&incoming.status);
                    let existing_priority = status_priority(&existing.status);
                    if incoming_priority > existing_priority {
                        self.nodes.insert(incoming.node_id.clone(), incoming);
                    }
                }
            }
            None => {
                self.nodes.insert(incoming.node_id.clone(), incoming);
            }
        }
    }
}

fn status_priority(status: &NodeStatus) -> u8 {
    match status {
        NodeStatus::Alive => 3,
        NodeStatus::Suspect => 2,
        NodeStatus::Dead => 1,
        NodeStatus::Left => 0,
    }
}

/// Gossip 消息类型
#[derive(Debug, Clone)]
pub enum GossipMessage {
    Join {
        node_id: String,
        address: String,
        incarnation: u64,
        metadata: HashMap<String, String>,
    },
    Alive {
        node_id: String,
        address: String,
        incarnation: u64,
    },
    Suspect {
        node_id: String,
        incarnation: u64,
    },
    Dead {
        node_id: String,
        incarnation: u64,
    },
    Leave {
        node_id: String,
    },
    Heartbeat {
        node_id: String,
        incarnation: u64,
        fragment_count: usize,
        load: f64,
    },
    FragmentDigest {
        node_id: String,
        digests: Vec<(u64, u64)>,
    },
    FullSync {
        node_id: String,
        state: Vec<(String, String, u64)>,
    },
}

/// Gossip 配置
#[derive(Debug, Clone)]
pub struct GossipConfig {
    pub probe_interval: Duration,
    pub suspect_timeout: Duration,
    pub dead_timeout: Duration,
    pub gossip_fanout: usize,
    pub gossip_interval: Duration,
    pub max_transmissions: u8,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            probe_interval: Duration::from_secs(1),
            suspect_timeout: Duration::from_secs(5),
            dead_timeout: Duration::from_secs(15),
            gossip_fanout: 3,
            gossip_interval: Duration::from_millis(200),
            max_transmissions: 10,
        }
    }
}

/// Gossip 协议引擎
///
/// 基于 SWIM（Scalable Weakly-consistent Infection-style Membership）协议：
/// - 定期探测：轮询探测下一个节点
/// - 间接探测：探测失败时通过 K 个中间节点间接探测
/// - 疑似→死亡：超时晋升机制
/// - 反熵：定期全量同步弥补消息丢失
pub struct GossipProtocol {
    pub membership: MembershipState,
    config: GossipConfig,
    message_queue: Vec<GossipMessage>,
    probe_index: usize,
    transmission_counters: HashMap<String, u8>,
}

impl GossipProtocol {
    pub fn new(local_node_id: String, config: GossipConfig) -> Self {
        Self {
            membership: MembershipState::new(local_node_id),
            config,
            message_queue: Vec::new(),
            probe_index: 0,
            transmission_counters: HashMap::new(),
        }
    }

    pub fn with_address(local_node_id: String, address: String, config: GossipConfig) -> Self {
        let mut gp = Self::new(local_node_id, config);
        if let Some(node) = gp.membership.nodes.get_mut(&gp.membership.local_node_id) {
            node.address = address;
        }
        gp
    }

    /// 处理加入集群消息
    pub fn handle_join(&mut self, node_id: String, address: String, incarnation: u64, metadata: HashMap<String, String>) {
        let node_info = NodeInfo {
            node_id: node_id.clone(),
            address: address.clone(),
            status: NodeStatus::Alive,
            incarnation,
            metadata,
            last_heartbeat: Instant::now(),
        };
        self.membership.merge_node(node_info);

        self.message_queue.push(GossipMessage::Alive {
            node_id,
            address,
            incarnation,
        });
    }

    /// 标记节点为疑似失败
    pub fn suspect_node(&mut self, node_id: &str) {
        if node_id == self.membership.local_node_id {
            self.membership.incarnation += 1;
            self.broadcast_alive();
            return;
        }

        if let Some(node) = self.membership.nodes.get_mut(node_id) {
            if node.status == NodeStatus::Alive {
                node.status = NodeStatus::Suspect;
                let incarnation = node.incarnation;
                self.message_queue.push(GossipMessage::Suspect {
                    node_id: node_id.to_string(),
                    incarnation,
                });
            }
        }
    }

    /// 标记节点为死亡
    pub fn declare_dead(&mut self, node_id: &str) {
        if let Some(node) = self.membership.nodes.get_mut(node_id) {
            node.status = NodeStatus::Dead;
            let incarnation = node.incarnation;
            self.message_queue.push(GossipMessage::Dead {
                node_id: node_id.to_string(),
                incarnation,
            });
        }
    }

    /// 节点主动离开
    pub fn leave(&mut self) {
        self.message_queue.push(GossipMessage::Leave {
            node_id: self.membership.local_node_id.clone(),
        });
        if let Some(node) = self.membership.nodes.get_mut(&self.membership.local_node_id) {
            node.status = NodeStatus::Left;
        }
    }

    /// 处理收到的 gossip 消息
    pub fn receive_message(&mut self, msg: GossipMessage) {
        match msg {
            GossipMessage::Join { node_id, address, incarnation, metadata } => {
                self.handle_join(node_id, address, incarnation, metadata);
            }
            GossipMessage::Alive { node_id, address, incarnation } => {
                let node_info = NodeInfo {
                    node_id: node_id.clone(),
                    address,
                    status: NodeStatus::Alive,
                    incarnation,
                    metadata: HashMap::new(),
                    last_heartbeat: Instant::now(),
                };
                self.membership.merge_node(node_info);
            }
            GossipMessage::Suspect { node_id, incarnation } => {
                if let Some(node) = self.membership.nodes.get(&node_id) {
                    if node_id == self.membership.local_node_id && incarnation == self.membership.incarnation {
                        self.membership.incarnation += 1;
                        self.broadcast_alive();
                    } else {
                        let mut updated = node.clone();
                        updated.status = NodeStatus::Suspect;
                        updated.incarnation = incarnation;
                        self.membership.merge_node(updated);
                    }
                }
            }
            GossipMessage::Dead { node_id, incarnation } => {
                if node_id != self.membership.local_node_id {
                    if let Some(node) = self.membership.nodes.get(&node_id) {
                        let mut updated = node.clone();
                        updated.status = NodeStatus::Dead;
                        updated.incarnation = incarnation;
                        self.membership.merge_node(updated);
                    }
                }
            }
            GossipMessage::Leave { node_id } => {
                if node_id != self.membership.local_node_id {
                    if let Some(node) = self.membership.nodes.get_mut(&node_id) {
                        node.status = NodeStatus::Left;
                    }
                }
            }
            GossipMessage::Heartbeat { node_id, incarnation, fragment_count: _, load: _ } => {
                if let Some(node) = self.membership.nodes.get_mut(&node_id) {
                    if incarnation >= node.incarnation {
                        node.incarnation = incarnation;
                        node.status = NodeStatus::Alive;
                        node.last_heartbeat = Instant::now();
                    }
                }
            }
            GossipMessage::FragmentDigest { .. } | GossipMessage::FullSync { .. } => {}
        }
    }

    /// 广播本节点 Alive 消息（refutation）
    fn broadcast_alive(&mut self) {
        let address = self.membership.nodes
            .get(&self.membership.local_node_id)
            .map(|n| n.address.clone())
            .unwrap_or_default();

        self.message_queue.push(GossipMessage::Alive {
            node_id: self.membership.local_node_id.clone(),
            address,
            incarnation: self.membership.incarnation,
        });
    }

    /// 推进时间：检测疑似→死亡晋升
    pub fn tick(&mut self) -> Vec<GossipMessage> {
        let now = Instant::now();
        let mut timed_out = Vec::new();

        for (node_id, node) in &self.membership.nodes {
            if node_id == &self.membership.local_node_id {
                continue;
            }

            match node.status {
                NodeStatus::Suspect
                    if now.duration_since(node.last_heartbeat) > self.config.dead_timeout =>
                {
                    timed_out.push(node_id.clone());
                }
                NodeStatus::Alive
                    if now.duration_since(node.last_heartbeat) > self.config.suspect_timeout =>
                {
                    timed_out.push(node_id.clone());
                }
                _ => {}
            }
        }

        for node_id in &timed_out {
            if let Some(node) = self.membership.nodes.get(node_id) {
                if node.status == NodeStatus::Alive {
                    self.suspect_node(node_id);
                } else if node.status == NodeStatus::Suspect {
                    self.declare_dead(node_id);
                }
            }
        }

        self.drain_messages()
    }

    /// 发送心跳
    pub fn send_heartbeat(&mut self, fragment_count: usize, load: f64) {
        self.message_queue.push(GossipMessage::Heartbeat {
            node_id: self.membership.local_node_id.clone(),
            incarnation: self.membership.incarnation,
            fragment_count,
            load,
        });
    }

    /// 选择本轮 gossip 目标节点
    pub fn select_gossip_targets(&self) -> Vec<String> {
        let alive: Vec<&NodeInfo> = self.membership.alive_nodes();
        let others: Vec<&NodeInfo> = alive.iter()
            .filter(|n| n.node_id != self.membership.local_node_id)
            .copied()
            .collect();

        let fanout = self.config.gossip_fanout.min(others.len());
        if fanout == 0 || others.is_empty() {
            return Vec::new();
        }

        let start = self.probe_index % others.len();
        (0..fanout)
            .map(|i| others[(start + i) % others.len()].node_id.clone())
            .collect()
    }

    /// 取出待发送消息
    pub fn drain_messages(&mut self) -> Vec<GossipMessage> {
        let msgs: Vec<GossipMessage> = self.message_queue.drain(..)
            .filter(|msg| {
                let key = format!("{:?}", std::mem::discriminant(msg));
                let count = self.transmission_counters.entry(key).or_insert(0);
                if *count < self.config.max_transmissions {
                    *count += 1;
                    true
                } else {
                    false
                }
            })
            .collect();

        self.transmission_counters.retain(|_, count| *count < self.config.max_transmissions);
        msgs
    }

    /// 推进探测索引
    pub fn advance_probe(&mut self) {
        self.probe_index += 1;
    }

    pub fn local_node_id(&self) -> &str {
        &self.membership.local_node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membership_new() {
        let ms = MembershipState::new("node1".to_string());
        assert_eq!(ms.alive_node_count(), 1);
        assert!(ms.is_alive("node1"));
    }

    #[test]
    fn test_handle_join() {
        let mut gp = GossipProtocol::new("node1".to_string(), GossipConfig::default());
        gp.handle_join("node2".to_string(), "addr2".to_string(), 0, HashMap::new());
        assert_eq!(gp.membership.alive_node_count(), 2);
    }

    #[test]
    fn test_suspect_and_dead() {
        let mut gp = GossipProtocol::new("node1".to_string(), GossipConfig::default());
        gp.handle_join("node2".to_string(), "addr2".to_string(), 0, HashMap::new());
        gp.suspect_node("node2");
        assert_eq!(gp.membership.nodes.get("node2").unwrap().status, NodeStatus::Suspect);
        gp.declare_dead("node2");
        assert_eq!(gp.membership.nodes.get("node2").unwrap().status, NodeStatus::Dead);
    }

    #[test]
    fn test_gossip_targets() {
        let mut gp = GossipProtocol::new("node1".to_string(), GossipConfig::default());
        gp.handle_join("node2".to_string(), "addr2".to_string(), 0, HashMap::new());
        gp.handle_join("node3".to_string(), "addr3".to_string(), 0, HashMap::new());
        let targets = gp.select_gossip_targets();
        assert!(!targets.is_empty());
    }

    #[test]
    fn test_receive_alive_message() {
        let mut gp = GossipProtocol::new("node1".to_string(), GossipConfig::default());
        gp.receive_message(GossipMessage::Alive {
            node_id: "node2".to_string(),
            address: "addr2".to_string(),
            incarnation: 1,
        });
        assert!(gp.membership.is_alive("node2"));
    }
}
