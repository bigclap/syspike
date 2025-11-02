//! Generic scheduler that coordinates graph steps and diffusion refinement.

use core_graph::{Network, StepObserver};
use tracing::{info, instrument};
use candle_core::Tensor;

use crate::diffusion::DiffusionLoop;

/// Configuration controlling how many discrete steps occur before diffusion.
#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    /// Number of discrete graph steps executed prior to diffusion.
    pub settle_steps: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { settle_steps: 3 }
    }
}

/// Result returned from a scheduler run.
#[derive(Debug, Clone)]
pub struct SchedulerOutcome {
    /// Final stabilised activation state.
    pub state: Vec<f32>,
    /// Number of diffusion iterations consumed.
    pub iterations: usize,
    /// Cosine similarity between the last two diffusion steps.
    pub similarity: f32,
    /// Aggregate energy of the network after diffusion.
    pub energy: f32,
}

/// Drives a [`Network`] through discrete steps followed by diffusion.
#[derive(Clone, Debug)]
pub struct ReasoningScheduler {
    config: SchedulerConfig,
}

impl ReasoningScheduler {
    /// Creates a scheduler with the provided configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self { config }
    }

    /// Executes a reasoning pass for the supplied embedding.
    #[instrument(skip(self, network, embedding, diffusion, observer))]
    pub fn run_case(
        &self,
        network: &mut Network,
        embedding: &Tensor,
        diffusion: &mut DiffusionLoop,
        mut observer: Option<&mut dyn StepObserver>,
    ) -> SchedulerOutcome {
        network.reset_state();
        network.inject_embedding(embedding);

        for step in 0..self.config.settle_steps {
            let step_span = tracing::info_span!("settle_step", step);
            let _entered = step_span.enter();
            let report = network.step(step);
            if let Some(hook) = observer.as_mut() {
                hook.on_step(step, network, &report);
            }
        }

        let state = diffusion.run(network).state;
        let energy = network.energy();
        let outcome = SchedulerOutcome {
            state,
            iterations: diffusion.last_iterations(),
            similarity: diffusion.last_similarity(),
            energy,
        };
        info!(
            iterations = outcome.iterations,
            similarity = outcome.similarity,
            energy = outcome.energy,
            "scheduler completed",
        );
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_graph::{ConnectionParams, NodeParams, StepObserver, assembly::GraphBuilder};
    use candle_core::Device;

    struct CountingObserver {
        invocations: usize,
    }

    impl CountingObserver {
        fn new() -> Self {
            Self { invocations: 0 }
        }
    }

    impl StepObserver for CountingObserver {
        fn on_step(&mut self, _step: usize, _network: &Network, _report: &core_graph::StepReport) {
            self.invocations += 1;
        }
    }

    #[test]
    fn scheduler_runs_steps_and_diffusion() {
        let mut builder = GraphBuilder::new();
        let input = builder.add_input_node(NodeParams::default());
        let output = builder.add_node(NodeParams::default());
        builder.add_connection(ConnectionParams::new(
            input, output, 1.0, 1.0, 0, 1.0, 1.0, 5.0, 5.0,
        ));
        let mut network = builder.build().expect("valid network");

        let mut diffusion = DiffusionLoop::new(crate::diffusion::DiffusionConfig {
            alpha_schedule: crate::diffusion::AnnealingSchedule::constant(0.5),
            sigma_schedule: crate::diffusion::AnnealingSchedule::constant(0.0),
            tolerance: 1e-3,
            jt_tolerance: 5e-4,
            stability_tolerance: 5e-4,
            stability_window: 2,
            max_energy_increase: usize::MAX,
            max_iters: 5,
            entropy_policy: crate::diffusion::EntropyPolicy::default(),
            fact_recruitment: None,
        });
        let scheduler = ReasoningScheduler::new(SchedulerConfig { settle_steps: 2 });

        let embedding = Tensor::from_vec(vec![1.0f32], (1,), &Device::Cpu).unwrap();
        let mut observer = CountingObserver::new();
        let outcome = scheduler.run_case(
            &mut network,
            &embedding,
            &mut diffusion,
            Some(&mut observer),
        );

        assert_eq!(observer.invocations, 2);
        assert_eq!(outcome.iterations, diffusion.last_iterations());
        assert_eq!(outcome.state.len(), network.nodes.len());
        assert!(outcome.energy >= 0.0);
    }
}
