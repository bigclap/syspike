//! Training pipeline scaffolding placeholder.

pub mod config;
pub mod metrics;
pub mod offline;
pub mod online;
pub mod optimizer;
pub mod mnist_autoencoder;

pub use config::OfflineTrainerConfig;
pub use metrics::{distinct_n, median_cosine_similarity};
pub use offline::{OfflineDecoderTrainer, ValidationRecord, ValidationReport};
pub use online::{OnlinePlasticity, OnlinePlasticityConfig, PlasticityStepOutcome, TraceLogEntry};
pub use optimizer::AdamWSchedule;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adamw_schedule_warmup_and_cosine() {
        let schedule = AdamWSchedule::new(1e-3, 1e-5, 2, 10);
        let step0 = schedule.learning_rate(0);
        let step2 = schedule.learning_rate(2);
        assert!(step0 > 0.0 && step0 < schedule.base_lr);
        assert!(step2 < schedule.base_lr);
    }

    #[test]
    fn offline_validation_reports_metrics() {
        let trainer = OfflineDecoderTrainer::new(OfflineTrainerConfig {
            validation_top_ks: vec![1, 2],
            ..OfflineTrainerConfig::default()
        });

        let record_one = ValidationRecord {
            predicted_embedding: vec![1.0, 0.0],
            target_embedding: vec![1.0, 0.0],
            retrieval_candidates: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generated_tokens: vec!["a".into(), "b".into()],
            target_tokens: vec!["a".into(), "b".into()],
            log_probs: vec![-0.1, -0.1],
        };

        let record_two = ValidationRecord {
            predicted_embedding: vec![0.0, 1.0],
            target_embedding: vec![1.0, 0.0],
            retrieval_candidates: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generated_tokens: vec!["a".into(), "c".into()],
            target_tokens: vec!["a".into(), "d".into()],
            log_probs: vec![-0.5, -1.0],
        };

        let report = trainer
            .validate(vec![record_one, record_two])
            .expect("validation report");

        assert!(report.cosine_at_median >= -1.0 && report.cosine_at_median <= 1.0);
        assert!(report.retrieval_rank_at_k.get(&1).is_some());
        assert!(report.distinct_n > 0.0);
        assert!(report.ppl_surrogate > 0.0);
    }

    #[test]
    fn online_plasticity_updates_on_interval() {
        let config = OnlinePlasticityConfig {
            update_interval: 2,
            structure_interval: 10,
            prune_threshold: 0.05,
            grow_threshold: 0.5,
            max_trace_log: 4,
        };
        let mut plasticity = OnlinePlasticity::new(vec![0.0, 0.0], config);

        let outcome_first = plasticity.step(vec![0.5, 1.0], 0.8).expect("first step");
        assert!(!outcome_first.applied_update);
        assert_eq!(plasticity.weights(), &[0.0, 0.0]);

        let outcome_second = plasticity.step(vec![0.5, 1.0], 0.8).expect("second step");
        assert!(outcome_second.applied_update);
        assert!(plasticity.weights()[1] > 0.0);
    }

    #[test]
    fn online_plasticity_prunes_and_grows_on_schedule() {
        let config = OnlinePlasticityConfig {
            update_interval: 1,
            structure_interval: 3,
            prune_threshold: 0.05,
            grow_threshold: 0.4,
            max_trace_log: 10,
        };
        let mut plasticity = OnlinePlasticity::new(vec![0.02, 0.3], config);

        plasticity.step(vec![0.01, 0.0], 0.2).unwrap();
        plasticity.step(vec![0.01, 0.0], 0.2).unwrap();
        let outcome = plasticity
            .step(vec![0.01, 0.0], 0.9)
            .expect("structure step");

        assert!(outcome.pruned > 0);
        assert!(outcome.grown > 0);
        assert!(plasticity.weights().len() >= 3);
    }

    #[test]
    fn online_plasticity_logs_recent_traces() {
        let config = OnlinePlasticityConfig {
            update_interval: 1,
            structure_interval: 5,
            prune_threshold: 0.1,
            grow_threshold: 0.5,
            max_trace_log: 2,
        };
        let mut plasticity = OnlinePlasticity::new(vec![0.0], config);

        plasticity.step(vec![0.1], 0.3).unwrap();
        plasticity.step(vec![0.2], 0.3).unwrap();
        plasticity.step(vec![0.3], 0.3).unwrap();

        let log = plasticity.trace_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].step, 2);
        assert_eq!(log[1].step, 3);
    }
}
