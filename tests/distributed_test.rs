use holographic_memory::*;

fn make_dist_config(node_id: &str, port: u16) -> DistributedConfig {
    DistributedConfig {
        node_id: node_id.to_string(),
        address: format!("127.0.0.1:{}", port),
        virtual_nodes: 150,
        replication_factor: 2,
        gossip_config: GossipConfig::default(),
        merkle_bucket_size: 256,
        max_concurrent_repairs: 2,
    }
}

#[test]
fn test_consistent_hash_ring_basic() {
    let mut ring = ConsistentHashRing::new(150);
    ring.add_node("node1").unwrap();
    ring.add_node("node2").unwrap();
    ring.add_node("node3").unwrap();

    assert_eq!(ring.node_count(), 3);
    assert_eq!(ring.virtual_node_count_total(), 450);

    let node = ring.find_node("test_key").unwrap();
    assert!(!node.is_empty());

    let replicas = ring.find_replicas("test_key", 2);
    assert!(replicas.len() <= 2);

    ring.remove_node("node2").unwrap();
    assert_eq!(ring.node_count(), 2);
}

#[test]
fn test_consistent_hash_load_balance() {
    let mut ring = ConsistentHashRing::new(150);
    ring.add_node("node1").unwrap();
    ring.add_node("node2").unwrap();
    ring.add_node("node3").unwrap();

    let dist = ring.load_distribution();
    let loads: Vec<usize> = dist.values().copied().collect();
    let avg = loads.iter().sum::<usize>() as f64 / loads.len() as f64;
    for &load in &loads {
        let deviation = (load as f64 - avg).abs() / avg;
        assert!(deviation < 0.3, "负载偏差过大: {} vs avg {}", load, avg);
    }
}

#[test]
fn test_gossip_membership() {
    let mut gp = GossipProtocol::new("node1".to_string(), GossipConfig::default());
    assert_eq!(gp.membership.alive_node_count(), 1);

    gp.handle_join("node2".to_string(), "addr2".to_string(), 0, std::collections::HashMap::new());
    gp.handle_join("node3".to_string(), "addr3".to_string(), 0, std::collections::HashMap::new());
    assert_eq!(gp.membership.alive_node_count(), 3);

    gp.suspect_node("node2");
    assert_eq!(gp.membership.nodes.get("node2").unwrap().status, NodeStatus::Suspect);

    gp.declare_dead("node2");
    assert_eq!(gp.membership.nodes.get("node2").unwrap().status, NodeStatus::Dead);
    assert_eq!(gp.membership.alive_node_count(), 2);
}

#[test]
fn test_gossip_message_exchange() {
    let mut gp1 = GossipProtocol::new("node1".to_string(), GossipConfig::default());
    let mut gp2 = GossipProtocol::new("node2".to_string(), GossipConfig::default());

    gp2.receive_message(GossipMessage::Alive {
        node_id: "node1".to_string(),
        address: "addr1".to_string(),
        incarnation: 0,
    });
    assert!(gp2.membership.is_alive("node1"));

    gp1.receive_message(GossipMessage::Alive {
        node_id: "node2".to_string(),
        address: "addr2".to_string(),
        incarnation: 0,
    });
    assert!(gp1.membership.is_alive("node2"));
}

#[test]
fn test_merkle_tree_diff_detection() {
    let digests1 = vec![(1, 100), (2, 200), (3, 300), (4, 400)];
    let digests2 = vec![(1, 100), (2, 999), (3, 300), (4, 400)];

    let mut tree1 = MerkleTree::new(2);
    let mut tree2 = MerkleTree::new(2);
    tree1.build_from_digests(&digests1);
    tree2.build_from_digests(&digests2);

    let report = tree1.diff(&tree2);
    assert!(!report.identical);
    assert!(!report.differing_ranges.is_empty());

    let mut tree3 = MerkleTree::new(2);
    tree3.build_from_digests(&digests1);
    let same_report = tree1.diff(&tree3);
    assert!(same_report.identical);
}

#[test]
fn test_anti_entropy_repair_flow() {
    let mut ae = AntiEntropyRepair::new(10, 2);
    ae.update_digests(vec![(1, 100), (2, 200), (3, 300)]);

    let local_hash = ae.root_hash();
    assert_ne!(local_hash, 0);

    let mut remote_tree = MerkleTree::new(10);
    remote_tree.build_from_digests(&[(1, 100), (2, 999), (3, 300)]);

    let diff = ae.compare_with(&remote_tree);
    assert!(!diff.identical);

    let task_id = ae.create_repair_task("local".to_string(), "remote".to_string(), diff.differing_ranges);
    ae.start_repair(task_id).unwrap();
    ae.complete_repair(task_id);
    assert_eq!(ae.completed_task_count(), 1);
}

#[test]
fn test_distributed_node_lifecycle() {
    let config = make_dist_config("node1", 7001);
    let mut node = DistributedHolographicNode::new(HolographicConfig::default(), config);

    node.join_cluster(&[
        ("node2".to_string(), "127.0.0.1:7002".to_string()),
    ]).unwrap();

    assert_eq!(node.ring.node_count(), 2);
    assert!(node.gossip.membership.is_alive("node2"));

    let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
    let result = node.distributed_store(&data).unwrap();
    assert!(result.quorum_achieved);

    let retrieved = node.distributed_retrieve(result.local_result.source_hash, data.len()).unwrap();
    assert!(!retrieved.from_replica);
    assert!(retrieved.integrity_verified);
}

#[test]
fn test_distributed_cluster_status() {
    let config = make_dist_config("node1", 7001);
    let mut node = DistributedHolographicNode::new(HolographicConfig::default(), config);
    node.ring.add_node("node1").unwrap();

    let status = node.cluster_status();
    assert_eq!(status.node_id, "node1");
    assert_eq!(status.total_nodes, 1);
    assert_eq!(status.replication_factor, 2);
}

#[test]
fn test_distributed_heartbeat_and_gossip() {
    let config = make_dist_config("node1", 7001);
    let mut node = DistributedHolographicNode::new(HolographicConfig::default(), config);
    node.ring.add_node("node1").unwrap();

    node.send_heartbeat();
    let msgs = node.gossip.drain_messages();
    assert!(!msgs.is_empty());

    let tick_msgs = node.gossip_tick();
    let _ = tick_msgs;
}

#[test]
fn test_distributed_multi_node_store() {
    let config1 = make_dist_config("node1", 7001);
    let config2 = DistributedConfig {
        replication_factor: 1,
        ..make_dist_config("node2", 7002)
    };

    let mut node1 = DistributedHolographicNode::new(HolographicConfig::default(), config1);
    let mut node2 = DistributedHolographicNode::new(HolographicConfig::default(), config2);

    node1.ring.add_node("node1").unwrap();
    node1.ring.add_node("node2").unwrap();
    node1.gossip.handle_join("node2".to_string(), "127.0.0.1:7002".to_string(), 0, std::collections::HashMap::new());

    node2.ring.add_node("node1").unwrap();
    node2.ring.add_node("node2").unwrap();

    let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).cos()).collect();

    let result1 = node1.distributed_store(&data).unwrap();
    assert!(result1.quorum_achieved);

    let result2 = node2.distributed_store(&data).unwrap();
    assert!(result2.quorum_achieved);

    let retrieved = node1.distributed_retrieve(result1.local_result.source_hash, data.len()).unwrap();
    assert!(!retrieved.from_replica);
}

#[test]
fn test_distributed_anti_entropy_trigger() {
    let config = DistributedConfig {
        replication_factor: 1,
        ..make_dist_config("node1", 7001)
    };
    let mut node = DistributedHolographicNode::new(HolographicConfig::default(), config);
    node.ring.add_node("node1").unwrap();

    let data: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
    node.distributed_store(&data).unwrap();

    let local_hash = node.anti_entropy.root_hash();
    let task_id = node.trigger_anti_entropy("node2", local_hash.wrapping_add(1));
    assert!(task_id.is_some());

    node.advance_repairs();
}
