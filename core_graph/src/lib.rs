//! Event-driven spiking network core with local plasticity helpers.

pub mod assembly;
pub mod io;
pub mod pools;
pub mod profiling;

pub use assembly::{AssemblyError, ConnectionParams, GraphBuilder, NodeParams};
pub use io::{GraphSnapshot, GraphSnapshotError};
pub use pools::{InhibitoryPoolConfig, RegionalDetectorConfig, RegionalDetectorState};
pub use profiling::{NetworkProfiler, ProfileSummary, ProfilerConfig, StepObserver, StepSnapshot};

use std::collections::HashMap;

use candle_core::Tensor;
use self::pools::{InhibitoryPoolRuntime, RegionalDetectorRuntime};
use serde::{Deserialize, Serialize};

/// Maximum discrete synaptic delay supported by the simulator.
pub const MAX_DELAY: usize = 16;
/// Alpha-kernel coefficients applied when delivering excitation.
pub const ALPHA_KERNEL: [f32; 8] = [0.6, 0.9, 0.8, 0.5, 0.3, 0.15, 0.08, 0.04];
/// Time window (in steps) used when estimating co-activation intensity.
const COACTIVATION_WINDOW: f32 = 8.0;

#[derive(Clone, Copy, Debug)]
/// Configuration for the structural plasticity maintenance pass.
pub struct StructuralPlasticityConfig {
    /// Multiplicative decay applied to the running co-activation counter each pass.
    pub coactivation_decay: f32,
    /// Minimum co-activation required to keep a connection alive when inactive.
    pub prune_threshold: f32,
    /// Number of consecutive steps without presynaptic spikes before pruning is considered.
    pub prune_inactivity_steps: usize,
    /// Co-activation required to regrow a previously pruned connection.
    pub growth_threshold: f32,
    /// Weight increment applied when re-activating a dormant connection.
    pub growth_increment: f32,
    /// Smallest permissible synaptic weight.
    pub weight_floor: f32,
    /// Learning rate that controls how aggressively delays follow the averaged latency.
    pub delay_learning_rate: f32,
    /// Minimum synaptic delay enforced during retuning.
    pub min_delay: usize,
    /// Maximum synaptic delay enforced during retuning.
    pub max_delay: usize,
}

impl Default for StructuralPlasticityConfig {
    fn default() -> Self {
        Self {
            coactivation_decay: 0.95,
            prune_threshold: 0.1,
            prune_inactivity_steps: 8,
            growth_threshold: 0.5,
            growth_increment: 0.05,
            weight_floor: 0.0,
            delay_learning_rate: 0.25,
            min_delay: 0,
            max_delay: MAX_DELAY - 1,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
/// Summary describing the structural plasticity adjustments performed in a pass.
pub struct StructuralChangeSummary {
    /// Number of connections that had their delay adjusted.
    pub retuned_delays: usize,
    /// Number of connections whose weight was pruned to the floor.
    pub pruned_connections: usize,
    /// Number of connections that were regrown after previously being pruned.
    pub regrown_connections: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// Functional category assigned to a [`Node`].
pub enum NodeType {
    /// Excitatory projection neuron delivering positive current.
    Excitatory,
    /// Inhibitory neuron providing divisive suppression.
    Inhibitory,
    /// Modulatory neuron temporarily adjusting downstream thresholds.
    Modulatory,
    /// Memory neuron with self-loop support for slow decay.
    Memory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// Policy applied when an episode boundary is reached.
pub enum EpisodeResetPolicy {
    /// No reset is performed when the boundary is reached.
    None,
    /// Clears fast-varying state while keeping adaptation and eligibility.
    Soft,
    /// Restores the node to its initial state.
    Hard,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Spiking unit with alpha-kernel synaptic integration and divisive normalisation.
pub struct Node {
    pub id: usize,
    pub node_type: NodeType,
    pub potential: f32,
    pub activation: f32,
    pub adaptation: f32,
    pub base_threshold: f32,
    pub lambda_v: f32,
    pub lambda_h: f32,
    pub activation_decay: f32,
    pub divisive_beta: f32,
    pub kappa: f32,
    pub modulation: f32,
    pub modulation_decay: f32,
    pub eligibility: f32,
    pub last_spike_step: Option<usize>,
    pub inhibition_accumulator: f32,
    pub future_exc: [f32; MAX_DELAY],
    pub accum_exc: f32,
    pub accum_inh: f32,
    pub energy_cap: f32,
    pub episode_period: Option<usize>,
    pub episode_phase: usize,
    pub reset_policy: EpisodeResetPolicy,
}

impl Node {
    /// Constructs a node with the supplied dynamics parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: usize,
        node_type: NodeType,
        base_threshold: f32,
        lambda_v: f32,
        lambda_h: f32,
        activation_decay: f32,
        divisive_beta: f32,
        kappa: f32,
        modulation_decay: f32,
        energy_cap: f32,
        episode_period: Option<usize>,
        reset_policy: EpisodeResetPolicy,
    ) -> Self {
        Self {
            id,
            node_type,
            potential: 0.0,
            activation: 0.0,
            adaptation: 0.0,
            base_threshold,
            lambda_v,
            lambda_h,
            activation_decay,
            divisive_beta,
            kappa,
            modulation: 0.0,
            modulation_decay,
            eligibility: 0.0,
            last_spike_step: None,
            inhibition_accumulator: 0.0,
            future_exc: [0.0; MAX_DELAY],
            accum_exc: 0.0,
            accum_inh: 0.0,
            energy_cap: energy_cap.max(f32::EPSILON),
            episode_period,
            episode_phase: 0,
            reset_policy,
        }
    }

    /// Returns the effective firing threshold after modulation and adaptation.
    pub fn effective_threshold(&self) -> f32 {
        let modulation_scale = (1.0 + self.modulation).clamp(0.1, 10.0);
        self.base_threshold * modulation_scale + self.adaptation
    }

    /// Integrates accumulated excitation and inhibition for one time step.
    pub fn step_integrate(&mut self) {
        self.advance_episode();

        let incoming_exc = self.future_exc[0];
        self.future_exc.rotate_left(1);
        self.future_exc[MAX_DELAY - 1] = 0.0;

        self.potential = self.potential * self.lambda_v + incoming_exc + self.accum_exc;
        self.adaptation *= self.lambda_h;
        self.activation *= self.activation_decay;
        self.modulation *= self.modulation_decay;

        let inhibition = self.accum_inh.max(0.0);
        if self.divisive_beta > 0.0 && (inhibition > 0.0 || self.adaptation > 0.0) {
            let denom = 1.0 + self.divisive_beta * (inhibition + self.adaptation.max(0.0));
            if denom > 0.0 {
                self.potential /= denom;
            }
        }
        self.inhibition_accumulator = inhibition;
        self.accum_exc = 0.0;
        self.accum_inh = 0.0;

        self.enforce_stability();
    }

    /// Applies spike side-effects such as resetting potential and recording time.
    pub fn on_spike(&mut self, step: usize) {
        self.potential = 0.0;
        self.activation = 1.0;
        self.adaptation += self.kappa;
        self.last_spike_step = Some(step);
        if matches!(self.node_type, NodeType::Memory) {
            let scaled = ((1.0 + self.modulation).clamp(0.1, 10.0)) * 0.9;
            self.modulation = (scaled - 1.0).clamp(-0.9, 9.0);
        }
        self.enforce_stability();
    }

    /// Clears transient state, restoring the node to its initial conditions.
    pub fn reset_state(&mut self) {
        self.potential = 0.0;
        self.activation = 0.0;
        self.adaptation = 0.0;
        self.modulation = 0.0;
        self.last_spike_step = None;
        self.eligibility = 0.0;
        self.inhibition_accumulator = 0.0;
        self.future_exc = [0.0; MAX_DELAY];
        self.accum_exc = 0.0;
        self.accum_inh = 0.0;
        self.episode_phase = 0;
        self.enforce_stability();
    }

    fn advance_episode(&mut self) {
        if let Some(period) = self.episode_period {
            if period == 0 {
                return;
            }
            self.episode_phase = self.episode_phase.saturating_add(1);
            if self.episode_phase >= period {
                match self.reset_policy {
                    EpisodeResetPolicy::None => {}
                    EpisodeResetPolicy::Soft => self.soft_reset(),
                    EpisodeResetPolicy::Hard => self.reset_state(),
                }
                if !matches!(self.reset_policy, EpisodeResetPolicy::Hard) {
                    self.episode_phase = 0;
                }
            }
        }
    }

    fn soft_reset(&mut self) {
        self.potential = 0.0;
        self.activation = 0.0;
        self.modulation = 0.0;
        self.last_spike_step = None;
        self.inhibition_accumulator = 0.0;
        self.future_exc = [0.0; MAX_DELAY];
        self.accum_exc = 0.0;
        self.accum_inh = 0.0;
        self.eligibility = self.eligibility.clamp(-self.energy_cap, self.energy_cap);
        self.episode_phase = 0;
        self.enforce_stability();
    }

    fn enforce_stability(&mut self) {
        if !self.potential.is_finite() {
            self.potential = 0.0;
        }
        if !self.activation.is_finite() {
            self.activation = 0.0;
        }
        if !self.adaptation.is_finite() {
            self.adaptation = 0.0;
        }
        if !self.modulation.is_finite() {
            self.modulation = 0.0;
        }
        if !self.eligibility.is_finite() {
            self.eligibility = 0.0;
        }
        if !self.inhibition_accumulator.is_finite() {
            self.inhibition_accumulator = 0.0;
        }
        if !self.accum_exc.is_finite() {
            self.accum_exc = 0.0;
        }
        if !self.accum_inh.is_finite() {
            self.accum_inh = 0.0;
        }

        let cap = self.energy_cap.max(f32::EPSILON);
        self.potential = self.potential.clamp(-cap, cap);
        self.activation = self.activation.clamp(0.0, cap);
        self.adaptation = self.adaptation.clamp(-cap, cap);
        self.modulation = self.modulation.clamp(-cap, cap);
        self.eligibility = self.eligibility.clamp(-cap, cap);
        self.inhibition_accumulator = self.inhibition_accumulator.clamp(0.0, cap);
        self.accum_exc = self.accum_exc.clamp(-cap, cap);
        self.accum_inh = self.accum_inh.clamp(-cap, cap);

        for slot in &mut self.future_exc {
            if !slot.is_finite() {
                *slot = 0.0;
            }
            *slot = slot.clamp(-cap, cap);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Directed synapse with reward-modulated STDP traces.
pub struct Connection {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
    pub max_weight: f32,
    pub delay: usize,
    pub eligibility: f32,
    pub elig_fast: f32,
    pub elig_slow: f32,
    pub dt_ema: f32,
    pub used_step: usize,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub last_pre_spike: Option<usize>,
    pub last_post_spike: Option<usize>,
    pub coactivation: f32,
    pub inactivity_steps: usize,
}

impl Connection {
    /// Creates a synapse with the provided STDP parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        from: usize,
        to: usize,
        weight: f32,
        max_weight: f32,
        delay: usize,
        a_plus: f32,
        a_minus: f32,
        tau_plus: f32,
        tau_minus: f32,
    ) -> Self {
        Self {
            from,
            to,
            weight,
            max_weight,
            delay,
            eligibility: 0.0,
            elig_fast: 0.0,
            elig_slow: 0.0,
            dt_ema: 0.0,
            used_step: 0,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            last_pre_spike: None,
            last_post_spike: None,
            coactivation: 0.0,
            inactivity_steps: 0,
        }
    }
}

#[derive(Default, Debug, Clone)]
/// Spike bookkeeping returned from [`Network::step`].
pub struct StepReport {
    pub spikes: Vec<usize>,
    pub modulatory_spikes: Vec<usize>,
}

/// Sparse event-driven network that hosts nodes and synapses.
pub struct Network {
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    input_nodes: Vec<usize>,
    inhibitory_pools: Vec<InhibitoryPoolRuntime>,
    regional_detectors: Vec<RegionalDetectorRuntime>,
    detector_lookup: HashMap<String, usize>,
}

impl Network {
    /// Builds a new network and pre-computes adjacency lists.
    pub fn new(nodes: Vec<Node>, connections: Vec<Connection>, input_nodes: Vec<usize>) -> Self {
        let mut outgoing = vec![Vec::new(); nodes.len()];
        let mut incoming = vec![Vec::new(); nodes.len()];
        for (idx, conn) in connections.iter().enumerate() {
            outgoing[conn.from].push(idx);
            incoming[conn.to].push(idx);
        }
        Self {
            nodes,
            connections,
            outgoing,
            incoming,
            input_nodes,
            inhibitory_pools: Vec::new(),
            regional_detectors: Vec::new(),
            detector_lookup: HashMap::new(),
        }
    }

    /// Resets all nodes and synapses to their default activation state.
    pub fn reset_state(&mut self) {
        for node in &mut self.nodes {
            node.reset_state();
        }
        for conn in &mut self.connections {
            conn.eligibility = 0.0;
            conn.elig_fast = 0.0;
            conn.elig_slow = 0.0;
            conn.dt_ema = 0.0;
            conn.used_step = 0;
            conn.last_post_spike = None;
            conn.last_pre_spike = None;
            conn.coactivation = 0.0;
            conn.inactivity_steps = 0;
        }
        for detector in &mut self.regional_detectors {
            detector.reset();
        }
    }

    /// Adds instantaneous excitation to the specified nodes.
    pub fn inject(&mut self, excitations: &[(usize, f32)]) {
        for &(id, value) in excitations {
            if let Some(node) = self.nodes.get_mut(id) {
                node.accum_exc += value;
            }
        }
    }

    /// Writes an embedding into the configured input interface.
    pub fn inject_embedding(&mut self, embedding: &Tensor) {
        let embedding: Vec<f32> = embedding.to_vec1().unwrap();
        for (node_id, value) in self.input_nodes.iter().zip(embedding.iter().copied()) {
            if let Some(node) = self.nodes.get_mut(*node_id) {
                node.accum_exc += value;
            }
        }
    }

    /// Configures inhibitory pools and optional regional detectors.
    pub fn configure_inhibitory_pools(
        &mut self,
        pools: Vec<InhibitoryPoolConfig>,
    ) -> Result<(), AssemblyError> {
        self.inhibitory_pools.clear();
        self.regional_detectors.clear();
        self.detector_lookup.clear();

        for pool in pools {
            let InhibitoryPoolConfig {
                members,
                inhibition_gain,
                detector,
            } = pool;
            let detector_index = if let Some(detector_cfg) = detector {
                if self
                    .detector_lookup
                    .contains_key(detector_cfg.label.as_str())
                {
                    return Err(AssemblyError::DuplicateDetectorLabel {
                        label: detector_cfg.label,
                    });
                }
                let index = self.regional_detectors.len();
                self.detector_lookup
                    .insert(detector_cfg.label.clone(), index);
                self.regional_detectors
                    .push(RegionalDetectorRuntime::new(detector_cfg));
                Some(index)
            } else {
                None
            };

            self.inhibitory_pools.push(InhibitoryPoolRuntime::new(
                members,
                inhibition_gain,
                detector_index,
            ));
        }

        Ok(())
    }

    /// Returns the state tracked for the detector with the supplied label.
    pub fn regional_detector_state(&self, label: &str) -> Option<RegionalDetectorState> {
        self.detector_lookup
            .get(label)
            .and_then(|&idx| self.regional_detectors.get(idx))
            .map(|detector| detector.state())
    }

    /// Returns snapshots for all configured regional detectors.
    pub fn regional_detector_states(&self) -> Vec<RegionalDetectorState> {
        self.regional_detectors
            .iter()
            .map(|detector| detector.state())
            .collect()
    }

    /// Returns the declarative configuration of all inhibitory pools.
    pub fn inhibitory_pool_configs(&self) -> Vec<InhibitoryPoolConfig> {
        self.inhibitory_pools
            .iter()
            .map(|pool| {
                let detector = pool.detector_index().and_then(|idx| {
                    self.regional_detectors
                        .get(idx)
                        .map(|runtime| RegionalDetectorConfig {
                            label: runtime.label().to_string(),
                            activation_threshold: runtime.activation_threshold(),
                            refresh_interval: runtime.refresh_interval(),
                        })
                });
                InhibitoryPoolConfig {
                    members: pool.members().to_vec(),
                    inhibition_gain: pool.inhibition_gain(),
                    detector,
                }
            })
            .collect()
    }

    /// Advances the network by one discrete step and returns fired nodes.
    pub fn step(&mut self, step: usize) -> StepReport {
        let mut report = StepReport::default();

        for conn in &mut self.connections {
            conn.inactivity_steps = conn.inactivity_steps.saturating_add(1);
        }

        for node in &mut self.nodes {
            node.step_integrate();
        }

        self.apply_inhibitory_pools(step);

        let modulatory_spikes: Vec<usize> = self
            .nodes
            .iter()
            .filter(|node| node.node_type == NodeType::Modulatory)
            .filter(|node| node.potential > node.effective_threshold())
            .map(|node| node.id)
            .collect();

        for node_id in modulatory_spikes.iter().copied() {
            let outgoing = self.outgoing[node_id].clone();
            for conn_id in outgoing {
                self.register_pre_spike(conn_id, step);
                let conn = &self.connections[conn_id];
                let weight = conn.weight;
                if let Some(target) = self.nodes.get_mut(conn.to) {
                    target.modulation += weight;
                    if target.modulation < -0.9 {
                        target.modulation = -0.9;
                    }
                }
            }
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.on_spike(step);
            }
            let incoming = self.incoming[node_id].clone();
            for conn_id in incoming {
                self.register_post_spike(conn_id, step);
            }
        }
        report.modulatory_spikes = modulatory_spikes;

        let mut spikes = Vec::new();
        for node in self.nodes.iter() {
            if node.node_type != NodeType::Modulatory && node.potential > node.effective_threshold()
            {
                spikes.push(node.id);
            }
        }

        for node_id in spikes.iter().copied() {
            let node_type = self.nodes[node_id].node_type;
            let outgoing = self.outgoing[node_id].clone();
            for conn_id in outgoing {
                self.register_pre_spike(conn_id, step);
                let conn = &self.connections[conn_id];
                match node_type {
                    NodeType::Inhibitory => {
                        if let Some(target) = self.nodes.get_mut(conn.to) {
                            schedule_inhibition(target, conn.weight, conn.delay);
                        }
                    }
                    _ => {
                        if let Some(target) = self.nodes.get_mut(conn.to) {
                            deliver(target, conn.weight, conn.delay);
                        }
                    }
                }
            }
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.on_spike(step);
            }
            let incoming = self.incoming[node_id].clone();
            for conn_id in incoming {
                self.register_post_spike(conn_id, step);
            }
        }
        report.spikes = spikes;

        report
    }

    /// Ratio of nodes with activation above the provided threshold.
    pub fn active_ratio(&self, activation_threshold: f32) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let active = self
            .nodes
            .iter()
            .filter(|node| node.activation > activation_threshold)
            .count();
        active as f32 / self.nodes.len() as f32
    }

    /// Applies a scalar reward to all synapses using the configured learning rate.
    pub fn apply_reward(&mut self, reward: f32, learning_rate: f32) {
        self.apply_reward_components(reward, reward, learning_rate);
    }

    /// Applies separate fast and slow rewards, updating traces and weights.
    pub fn apply_reward_components(
        &mut self,
        reward_fast: f32,
        reward_slow: f32,
        learning_rate: f32,
    ) {
        for conn in &mut self.connections {
            let delta_w =
                learning_rate * (reward_fast * conn.elig_fast + reward_slow * conn.elig_slow);
            if delta_w != 0.0 {
                conn.weight = (conn.weight + delta_w).clamp(0.0, conn.max_weight);
            }
            conn.eligibility = conn.elig_fast + conn.elig_slow;
            conn.elig_fast *= 0.98;
            conn.elig_slow *= 0.999;
            conn.used_step = 0;
        }
    }

    /// Returns the weight of a connection by index or zero if missing.
    pub fn connection_weight(&self, connection_id: usize) -> f32 {
        self.connections
            .get(connection_id)
            .map(|conn| conn.weight)
            .unwrap_or(0.0)
    }

    /// Returns the fast and slow eligibility traces for a connection.
    pub fn connection_traces(&self, connection_id: usize) -> (f32, f32) {
        self.connections
            .get(connection_id)
            .map(|conn| (conn.elig_fast, conn.elig_slow))
            .unwrap_or((0.0, 0.0))
    }

    fn apply_inhibitory_pools(&mut self, step: usize) {
        for idx in 0..self.inhibitory_pools.len() {
            let (winner, peak) = {
                let pool = &self.inhibitory_pools[idx];
                pool.determine_winner(&self.nodes)
            };

            if let Some(winner_id) = winner {
                let members = self.inhibitory_pools[idx].members().to_vec();
                let gain = self.inhibitory_pools[idx].inhibition_gain().max(0.0);
                if gain > 0.0 {
                    for node_id in members {
                        if node_id == winner_id {
                            continue;
                        }
                        if let Some(node) = self.nodes.get_mut(node_id) {
                            if node.potential.is_finite() {
                                node.potential -= gain;
                                if !node.potential.is_finite() {
                                    node.potential = 0.0;
                                }
                            }
                            node.inhibition_accumulator += gain;
                        }
                    }
                }
            }

            if let Some(detector_idx) = self.inhibitory_pools[idx].detector_index() {
                if let Some(detector) = self.regional_detectors.get_mut(detector_idx) {
                    detector.observe(step, winner, peak);
                }
            }
        }
    }

    /// Updates STDP traces for a presynaptic spike event.
    fn register_pre_spike(&mut self, conn_id: usize, step: usize) {
        let to = self.connections[conn_id].to;
        let dst_last_spike = self.nodes[to].last_spike_step;
        let conn = &mut self.connections[conn_id];
        let dt = dst_last_spike.map(|last| step as f32 - last as f32);
        if let Some(dt) = dt {
            let tau = conn.tau_minus.max(1e-6);
            let decay = (-(dt) / tau).exp();
            conn.elig_fast -= conn.a_minus * decay;
            conn.elig_slow -= 0.1 * conn.a_minus * decay;
            conn.dt_ema = 0.85 * conn.dt_ema + 0.15 * dt;
            accumulate_coactivation(conn, dt);
        } else {
            conn.dt_ema *= 0.85;
        }
        conn.last_pre_spike = Some(step);
        conn.used_step = step;
        conn.inactivity_steps = 0;
    }

    /// Updates STDP traces for a postsynaptic spike event.
    fn register_post_spike(&mut self, conn_id: usize, step: usize) {
        let pre_last = self.connections[conn_id].last_pre_spike;
        let conn = &mut self.connections[conn_id];
        if let Some(pre_last) = pre_last {
            let dt = step as f32 - pre_last as f32;
            if dt >= 0.0 {
                let tau = conn.tau_plus.max(1e-6);
                let factor = (-(dt) / tau).exp();
                conn.elig_fast += conn.a_plus * factor;
                conn.elig_slow += 0.1 * conn.a_plus * factor;
                conn.dt_ema = 0.85 * conn.dt_ema + 0.15 * dt;
                accumulate_coactivation(conn, dt);
            }
        } else {
            conn.dt_ema *= 0.85;
        }
        conn.last_post_spike = Some(step);
        conn.used_step = step;
        conn.inactivity_steps = 0;
    }

    /// Immutable access to a node by identifier.
    pub fn node(&self, node_id: usize) -> &Node {
        &self.nodes[node_id]
    }

    /// Mutable access to a node by identifier.
    pub fn node_mut(&mut self, node_id: usize) -> &mut Node {
        &mut self.nodes[node_id]
    }

    /// Collects current activation values into a contiguous vector.
    pub fn state_vector(&self) -> Vec<f32> {
        self.nodes.iter().map(|node| node.activation).collect()
    }

    /// Overwrites node activations with values from the supplied slice.
    pub fn set_state(&mut self, state: &[f32]) {
        for (node, value) in self.nodes.iter_mut().zip(state.iter().copied()) {
            node.activation = value;
        }
    }

    /// Computes the consensus activation state based on incoming connections.
    pub fn consensus_state(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.nodes.len());
        for (idx, node) in self.nodes.iter().enumerate() {
            let incoming = &self.incoming[idx];
            if incoming.is_empty() {
                result.push(node.activation);
                continue;
            }
            let mut sum = 0.0;
            let mut total_weight = 0.0;
            for &conn_id in incoming {
                let conn = &self.connections[conn_id];
                let source = &self.nodes[conn.from];
                let strength = conn.weight.abs() * source.activation.abs();
                if strength == 0.0 {
                    continue;
                }
                let sign = match source.node_type {
                    NodeType::Inhibitory | NodeType::Modulatory => -1.0,
                    NodeType::Excitatory | NodeType::Memory => 1.0,
                };
                sum += sign * conn.weight * source.activation;
                total_weight += strength;
            }
            if total_weight == 0.0 {
                result.push(node.activation);
            } else {
                let average = (node.activation + sum) / (1.0 + total_weight);
                result.push(average);
            }
        }
        result
    }

    /// Lightweight energy proxy used for runtime diagnostics.
    pub fn energy(&self) -> f32 {
        self.nodes
            .iter()
            .map(|node| node.potential.abs() + node.activation)
            .sum()
    }

    /// Returns the number of active nodes and edges given an activation threshold.
    pub fn active_counts(&self, activation_threshold: f32) -> (usize, usize) {
        if self.nodes.is_empty() {
            return (0, 0);
        }
        let mut active_mask = vec![false; self.nodes.len()];
        let mut active_nodes = 0usize;
        for (idx, node) in self.nodes.iter().enumerate() {
            if node.activation >= activation_threshold {
                active_mask[idx] = true;
                active_nodes += 1;
            }
        }
        let mut active_edges = 0usize;
        for conn in &self.connections {
            if conn.weight.abs() <= f32::EPSILON {
                continue;
            }
            if active_mask.get(conn.from).copied().unwrap_or(false)
                && active_mask.get(conn.to).copied().unwrap_or(false)
            {
                active_edges += 1;
            }
        }
        (active_nodes, active_edges)
    }

    /// Rough estimate of heap fragmentation for the core network structures.
    pub fn memory_fragmentation(&self) -> f32 {
        fn fragmentation(len: usize, capacity: usize) -> f32 {
            if capacity == 0 {
                return 0.0;
            }
            let utilisation = len as f32 / capacity as f32;
            (1.0 - utilisation).clamp(0.0, 1.0)
        }
        let node_fragmentation = fragmentation(self.nodes.len(), self.nodes.capacity());
        let connection_fragmentation =
            fragmentation(self.connections.len(), self.connections.capacity());
        (node_fragmentation + connection_fragmentation) * 0.5
    }

    /// Identifiers for nodes that receive embedding inputs.
    pub fn input_nodes(&self) -> &[usize] {
        &self.input_nodes
    }

    /// Applies structural plasticity rules over all connections.
    pub fn apply_structural_plasticity(
        &mut self,
        config: &StructuralPlasticityConfig,
    ) -> StructuralChangeSummary {
        let mut summary = StructuralChangeSummary::default();
        for conn in &mut self.connections {
            conn.coactivation *= config.coactivation_decay.clamp(0.0, 1.0);

            if conn.weight <= config.weight_floor + f32::EPSILON
                && conn.coactivation > config.growth_threshold
            {
                let new_weight = (conn.weight + config.growth_increment).min(conn.max_weight);
                if new_weight > conn.weight {
                    conn.weight = new_weight;
                    summary.regrown_connections += 1;
                    conn.inactivity_steps = 0;
                    conn.coactivation = 0.0;
                }
            }

            if conn.inactivity_steps >= config.prune_inactivity_steps
                && conn.coactivation < config.prune_threshold
                && conn.weight > config.weight_floor + f32::EPSILON
            {
                conn.weight = config.weight_floor;
                summary.pruned_connections += 1;
                conn.coactivation = 0.0;
            }

            let target_delay = conn
                .dt_ema
                .max(config.min_delay as f32)
                .min(config.max_delay as f32);
            if target_delay.is_finite() {
                let delay_float = conn.delay as f32;
                let updated = delay_float
                    + config.delay_learning_rate.clamp(0.0, 1.0) * (target_delay - delay_float);
                let new_delay = updated.round() as usize;
                let new_delay = new_delay.clamp(config.min_delay, config.max_delay);
                if new_delay != conn.delay {
                    conn.delay = new_delay;
                    summary.retuned_delays += 1;
                }
            }
        }
        summary
    }
}

/// Schedules delayed excitation delivery using the alpha kernel.
fn deliver(node: &mut Node, weight: f32, delay: usize) {
    let start = delay.min(MAX_DELAY - 1);
    for (idx, coeff) in ALPHA_KERNEL.iter().enumerate() {
        let slot = start + idx;
        if slot >= MAX_DELAY {
            break;
        }
        node.future_exc[slot] += weight * coeff;
    }
}

/// Accumulates inhibition for divisive normalisation.
fn schedule_inhibition(node: &mut Node, weight: f32, _delay: usize) {
    node.accum_inh += weight.abs();
}

fn accumulate_coactivation(conn: &mut Connection, dt: f32) {
    if !dt.is_finite() {
        return;
    }
    let magnitude = (COACTIVATION_WINDOW - dt.abs()).max(0.0) / COACTIVATION_WINDOW;
    if magnitude > 0.0 {
        conn.coactivation += magnitude;
    }
}

pub mod test_helpers {
    use super::{Connection, EpisodeResetPolicy, Network, Node, NodeType};

    pub fn two_node_modulatory() -> Network {
        let nodes = vec![
            Node::new(
                0,
                NodeType::Modulatory,
                0.5,
                0.99,
                0.9,
                0.9,
                0.0,
                0.05,
                0.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
            Node::new(
                1,
                NodeType::Excitatory,
                0.6,
                0.99,
                0.9,
                0.9,
                0.0,
                0.05,
                0.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
        ];
        let connections = vec![Connection::new(0, 1, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0)];
        Network::new(nodes, connections, vec![0])
    }

    pub fn simple_pair() -> Network {
        let nodes = vec![
            Node::new(
                0,
                NodeType::Excitatory,
                0.3,
                0.99,
                0.9,
                0.9,
                0.0,
                0.05,
                0.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
            Node::new(
                1,
                NodeType::Excitatory,
                0.5,
                0.99,
                0.9,
                0.9,
                0.0,
                0.05,
                0.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
        ];
        let connections = vec![Connection::new(0, 1, 0.8, 2.0, 0, 1.0, 1.0, 5.0, 5.0)];
        Network::new(nodes, connections, vec![0])
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ALPHA_KERNEL, Connection, EpisodeResetPolicy, Network, Node, NodeType,
        StructuralPlasticityConfig, test_helpers,
    };

    fn plasticity_config_for_tests() -> StructuralPlasticityConfig {
        StructuralPlasticityConfig {
            prune_inactivity_steps: 3,
            prune_threshold: 0.2,
            growth_threshold: 0.4,
            growth_increment: 0.2,
            delay_learning_rate: 0.5,
            ..StructuralPlasticityConfig::default()
        }
    }

    fn stimulate_pair(network: &mut Network, start_step: usize) {
        network.inject(&[(0, 1.0)]);
        let report_pre = network.step(start_step);
        assert!(report_pre.spikes.contains(&0));

        network.inject(&[(1, 0.4)]);
        let report_post = network.step(start_step + 1);
        assert!(report_post.spikes.contains(&1));
    }

    #[test]
    fn inhibitory_spikes_apply_divisive_normalization() {
        let nodes = vec![
            Node::new(
                0,
                NodeType::Excitatory,
                0.3,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
            Node::new(
                1,
                NodeType::Inhibitory,
                0.3,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
            Node::new(
                2,
                NodeType::Excitatory,
                2.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.0,
                0.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
        ];
        let connections = vec![
            Connection::new(0, 2, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0),
            Connection::new(1, 2, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0),
        ];
        let mut network = Network::new(nodes, connections, vec![0, 1]);

        network.inject(&[(0, 1.0), (1, 1.0)]);
        let _ = network.step(0);
        let _ = network.step(1);

        let target = network.node(2);
        let expected = ALPHA_KERNEL[0] / (1.0 + 0.5 * 1.0);
        assert!((target.potential - expected).abs() < 1e-6);
        assert!(target.activation < 1.0);
    }

    #[test]
    fn memory_nodes_self_loop_retain_potential() {
        let nodes = vec![Node::new(
            0,
            NodeType::Memory,
            0.5,
            0.99,
            0.995,
            0.98,
            0.0,
            0.0,
            0.98,
            10.0,
            None,
            EpisodeResetPolicy::None,
        )];
        let connections = vec![Connection::new(0, 0, 0.4, 1.5, 0, 1.0, 1.0, 5.0, 5.0)];
        let mut network = Network::new(nodes, connections, vec![]);

        network.inject(&[(0, 1.0)]);
        let first = network.step(0);
        assert!(first.spikes.contains(&0));

        let after_spike_threshold = network.node(0).effective_threshold();
        assert!(after_spike_threshold < 0.5);

        let _ = network.step(1);
        let potential_after_one = network.node(0).potential;
        let expected_one = ALPHA_KERNEL[0] * 0.4;
        assert!((potential_after_one - expected_one).abs() < 1e-6);

        let report_two = network.step(2);
        assert!(report_two.spikes.contains(&0));
        let potential_after_two = network.node(0).potential;
        assert!(potential_after_two.abs() < 1e-6);

        let report_three = network.step(3);
        assert!(report_three.spikes.contains(&0));
    }

    #[test]
    fn memory_nodes_recover_threshold_after_inactivity() {
        let nodes = vec![Node::new(
            0,
            NodeType::Memory,
            0.6,
            0.99,
            0.99,
            0.98,
            0.0,
            0.0,
            0.7,
            10.0,
            None,
            EpisodeResetPolicy::None,
        )];
        let connections = vec![Connection::new(0, 0, 0.12, 1.0, 0, 1.0, 1.0, 5.0, 5.0)];
        let mut network = Network::new(nodes, connections, vec![]);

        network.inject(&[(0, 1.0)]);
        let _ = network.step(0);
        let lowered_threshold = network.node(0).effective_threshold();
        assert!(lowered_threshold < 0.6);

        for step in 1..=6 {
            let _ = network.step(step);
        }

        let recovered_threshold = network.node(0).effective_threshold();
        assert!(recovered_threshold > 0.58);
        assert!(recovered_threshold < 0.62);
    }

    #[test]
    fn excitatory_spike_delivers_alpha_kernel_over_time() {
        let nodes = vec![
            Node::new(
                0,
                NodeType::Excitatory,
                0.2,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
            Node::new(
                1,
                NodeType::Excitatory,
                10.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                10.0,
                None,
                EpisodeResetPolicy::None,
            ),
        ];
        let connections = vec![Connection::new(0, 1, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0)];
        let mut network = Network::new(nodes, connections, vec![0]);

        network.inject(&[(0, 5.0)]);
        let first = network.step(0);
        assert!(first.spikes.contains(&0));

        for (step, expected) in ALPHA_KERNEL.iter().copied().enumerate() {
            let report = network.step(step + 1);
            assert!(report.spikes.is_empty());
            let potential = network.node(1).potential;
            assert!(
                (potential - expected).abs() < 1e-6,
                "step {step}: expected {expected}, got {potential}"
            );
        }
    }

    #[test]
    fn eligibility_traces_update_and_decay() {
        let mut network = test_helpers::simple_pair();

        network.inject(&[(0, 1.0)]);
        let _ = network.step(0);
        network.inject(&[(1, 1.0)]);
        let report = network.step(1);
        assert!(report.spikes.contains(&1));

        let (fast_before, slow_before) = network.connection_traces(0);
        assert!(
            fast_before > 0.0,
            "fast trace should increase after causal pair"
        );
        assert!(slow_before > 0.0 && slow_before < fast_before);

        let weight_before = network.connection_weight(0);
        network.apply_reward_components(1.0, 0.5, 0.1);
        let weight_after = network.connection_weight(0);
        let expected_delta = 0.1 * (1.0 * fast_before + 0.5 * slow_before);
        assert!(
            (weight_after - (weight_before + expected_delta)).abs() < 1e-5,
            "unexpected weight update"
        );

        let (fast_after, slow_after) = network.connection_traces(0);
        assert!((fast_after - fast_before * 0.98).abs() < 1e-6);
        assert!((slow_after - slow_before * 0.999).abs() < 1e-6);
    }

    #[test]
    fn coactivation_accumulates_for_spike_pairs() {
        let mut network = test_helpers::simple_pair();
        network.node_mut(1).base_threshold = 0.2;

        stimulate_pair(&mut network, 0);

        let coactivation = network.connections[0].coactivation;
        assert!(
            coactivation > 0.0,
            "coactivation should increase after paired spikes"
        );
    }

    #[test]
    fn structural_plasticity_prunes_inactive_connections() {
        let mut network = test_helpers::simple_pair();
        network.connections[0].weight = 0.6;
        network.connections[0].max_weight = 1.0;
        let config = plasticity_config_for_tests();

        for step in 0..=4 {
            let _ = network.step(step);
        }

        let summary = network.apply_structural_plasticity(&config);
        assert_eq!(summary.pruned_connections, 1);
        assert!(network.connections[0].weight <= config.weight_floor + 1e-6);
    }

    #[test]
    fn energy_cap_limits_state_growth() {
        let mut node = Node::new(
            0,
            NodeType::Excitatory,
            0.1,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.5,
            None,
            EpisodeResetPolicy::None,
        );

        node.accum_exc = 10.0;
        node.step_integrate();
        assert!(node.potential <= 0.5 + 1e-6);
        assert!(node.potential >= -0.5 - 1e-6);

        node.accum_exc = -10.0;
        node.step_integrate();
        assert!(node.potential >= -0.5 - 1e-6);
        assert!(node.potential <= 0.5 + 1e-6);
    }

    #[test]
    fn nan_states_are_sanitised() {
        let mut node = Node::new(
            0,
            NodeType::Excitatory,
            0.1,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            None,
            EpisodeResetPolicy::None,
        );

        node.potential = f32::NAN;
        node.activation = f32::NAN;
        node.adaptation = f32::NAN;
        node.modulation = f32::NAN;
        node.eligibility = f32::NAN;
        node.inhibition_accumulator = f32::NAN;
        node.accum_exc = f32::NAN;
        node.accum_inh = f32::NAN;
        node.future_exc[0] = f32::NAN;

        node.step_integrate();

        assert_eq!(node.potential, 0.0);
        assert_eq!(node.activation, 0.0);
        assert_eq!(node.adaptation, 0.0);
        assert_eq!(node.modulation, 0.0);
        assert!(node.future_exc.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn episodic_reset_policy_resets_state() {
        let mut node = Node::new(
            0,
            NodeType::Excitatory,
            0.1,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            Some(2),
            EpisodeResetPolicy::Hard,
        );

        node.accum_exc = 1.0;
        node.step_integrate();
        assert!(node.potential > 0.0);

        node.accum_exc = 1.0;
        node.step_integrate();
        assert_eq!(node.potential, 0.0);

        node.accum_exc = 1.0;
        node.step_integrate();
        assert!(node.potential > 0.0);
    }

    #[test]
    fn structural_plasticity_regrows_after_activity() {
        let mut network = test_helpers::simple_pair();
        network.node_mut(1).base_threshold = 0.2;
        let config = plasticity_config_for_tests();

        for step in 0..=4 {
            let _ = network.step(step);
        }
        let _ = network.apply_structural_plasticity(&config);
        assert!(network.connections[0].weight <= config.weight_floor + 1e-6);

        let mut step = 5;
        for _ in 0..3 {
            stimulate_pair(&mut network, step);
            step += 2;
        }

        network.connections[0].coactivation = 1.0; // ensure above threshold
        let summary = network.apply_structural_plasticity(&config);
        assert_eq!(summary.regrown_connections, 1);
        assert!(network.connections[0].weight > config.weight_floor);
    }

    #[test]
    fn structural_plasticity_retunes_delay_towards_latency_estimate() {
        let mut network = test_helpers::simple_pair();
        let mut config = StructuralPlasticityConfig {
            delay_learning_rate: 0.5,
            prune_inactivity_steps: usize::MAX,
            ..StructuralPlasticityConfig::default()
        };
        config.prune_threshold = -1.0;
        config.growth_threshold = 100.0;

        network.connections[0].delay = 1;
        network.connections[0].dt_ema = 5.0;

        let summary = network.apply_structural_plasticity(&config);
        assert_eq!(summary.retuned_delays, 1);
        assert_eq!(network.connections[0].delay, 3);
    }
}
