use super::*;

fn base_config() -> RetrieverConfig {
    RetrieverConfig {
        dimension: 2,
        value_dimension: 2,
        max_elements: 16,
        max_layers: 8,
        max_connections: 4,
        ef_construction: 16,
        ef_search: 16,
        top_k: 3,
        gate_decay: 1.0,
        gate_floor: 0.0,
        gate_ceiling: 1.0,
        gate_refresh: 1.0,
    }
}

fn record(key: u64, embedding: [f32; 2]) -> MemoryRecord {
    MemoryRecord::new(key, embedding.to_vec(), embedding.to_vec())
}

#[test]
fn ann_search_returns_ranked_matches() {
    let mut retriever = Retriever::new(base_config()).expect("valid config");
    retriever
        .ingest([
            record(1, [1.0, 0.1]),
            record(2, [0.1, 1.0]),
            record(3, [0.7, 0.7]),
        ])
        .expect("ingest succeeds");

    let hits = retriever
        .search(&[1.0, 0.0], Some(2))
        .expect("query executes");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].key, 1);
    assert!(hits[0].similarity > hits[1].similarity);
    assert!(hits[0].similarity > 0.9);
}

#[test]
fn snapshot_round_trip_preserves_results() {
    let mut retriever = Retriever::new(base_config()).expect("valid config");
    retriever
        .ingest([record(7, [1.0, 0.0]), record(8, [0.0, 1.0])])
        .expect("ingest succeeds");
    let snapshot = retriever.snapshot().expect("snapshot");

    let mut restored = Retriever::from_snapshot(base_config(), snapshot).expect("restored");
    let hits = restored.search(&[0.0, 1.0], None).expect("query executes");
    assert_eq!(hits.first().map(|hit| hit.key), Some(8));
    let value = restored.value(8).expect("payload available");
    assert!((value[1] - 1.0).abs() < 1e-6);
}

#[test]
fn snapshot_restores_gate_state() {
    let mut config = base_config();
    config.gate_decay = 0.5;
    config.gate_refresh = 1.0;
    config.gate_floor = 0.1;
    config.gate_ceiling = 1.0;
    let mut retriever = Retriever::new(config.clone()).expect("valid config");
    retriever
        .ingest([record(1, [1.0, 0.0]), record(2, [0.0, 1.0])])
        .expect("ingest succeeds");

    retriever
        .search(&[1.0, 0.0], Some(1))
        .expect("query executes");

    let gate_one = retriever.gate(1).expect("gate available");
    let gate_two = retriever.gate(2).expect("gate available");
    assert!((gate_one - 1.0).abs() < 1e-6);
    assert!((gate_two - 0.5).abs() < 1e-6);

    let snapshot = retriever.snapshot().expect("snapshot");
    let restored = Retriever::from_snapshot(config, snapshot).expect("restored");

    let restored_gate_one = restored.gate(1).expect("gate available");
    let restored_gate_two = restored.gate(2).expect("gate available");
    assert!((restored_gate_one - gate_one).abs() < 1e-6);
    assert!((restored_gate_two - gate_two).abs() < 1e-6);
}

#[test]
fn refresh_pulses_decay_and_boost_gates() {
    let mut config = base_config();
    config.gate_decay = 0.9;
    config.gate_floor = 0.1;
    config.gate_ceiling = 1.0;
    config.gate_refresh = 0.95;
    let mut retriever = Retriever::new(config).expect("valid config");
    retriever
        .ingest([record(1, [1.0, 0.0]), record(2, [0.0, 1.0])])
        .expect("ingest succeeds");

    assert!((retriever.gate(1).unwrap() - 0.95).abs() < 1e-6);
    assert!((retriever.gate(2).unwrap() - 0.95).abs() < 1e-6);

    let hits = retriever
        .search(&[1.0, 0.0], Some(1))
        .expect("query executes");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, 1);
    assert!((hits[0].gate - 0.855).abs() < 1e-6);
    assert!((retriever.gate(1).unwrap() - 0.95).abs() < 1e-6);
    assert!((retriever.gate(2).unwrap() - 0.855).abs() < 1e-6);

    retriever.refresh_pulse(&[]);
    retriever.refresh_pulse(&[]);
    let degraded_gate = retriever.gate(2).unwrap();
    assert!(degraded_gate < 0.8);

    let hits = retriever
        .search(&[0.0, 1.0], Some(1))
        .expect("query executes");
    assert_eq!(hits[0].key, 2);
    assert!(hits[0].gate < degraded_gate);
    assert!(retriever.gate(2).unwrap() > 0.9);
    assert!(retriever.gate(1).unwrap() < 0.8);
}
