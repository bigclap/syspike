//! Image autoencoder demo wiring.

use anyhow::Result;
use candle_core::{Device, Tensor};
use core_graph::{
    NetworkProfiler, ProfilerConfig, Network, ConnectionParams, EpisodeResetPolicy, GraphBuilder,
    NodeParams, NodeType,
};
use core_rules::{
    diffusion::{AnnealingSchedule, DiffusionConfig, DiffusionLoop, EntropyPolicy},
    scheduler::{ReasoningScheduler, SchedulerConfig},
};
use model_image_enc::{FrozenImageEncoder, Image};
use model_image_dec::RawPixelDecoder;

/// Builds a simple image autoencoder network.
pub fn build_image_autoencoder_network(
    width: u16,
    height: u16,
    device: &Device,
) -> (Network, FrozenImageEncoder, RawPixelDecoder, Vec<usize>, Vec<usize>) {
    let mut builder = GraphBuilder::new();
    let pixel_count = (width * height) as usize;

    let mut input_nodes = Vec::with_capacity(pixel_count * 6);
    for _ in 0..(pixel_count * 6) {
        input_nodes.push(builder.add_input_node(NodeParams::new(
            NodeType::Excitatory,
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, None, EpisodeResetPolicy::None,
        )));
    }

    let mut output_nodes = Vec::with_capacity(pixel_count * 6);
    for _ in 0..(pixel_count * 6) {
        output_nodes.push(builder.add_node(NodeParams::new(
            NodeType::Excitatory,
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, None, EpisodeResetPolicy::None,
        )));
    }

    for i in 0..(pixel_count * 6) {
        builder.add_connection(ConnectionParams::new(
            input_nodes[i],
            output_nodes[i],
            1.0, 1.0, 0, 1.0, 1.0, 1.0, 1.0,
        ));
    }

    let network = builder.build().expect("image autoencoder network assembly");

    let encoder = FrozenImageEncoder::new(width, height, device.clone()).unwrap();
    let decoder = RawPixelDecoder::new(width, height);

    (network, encoder, decoder, input_nodes, output_nodes)
}

pub fn run_image_autoencoder_inference(
    image: &Image,
    device: &Device,
) -> Result<Image> {
    let (mut network, encoder, decoder, _input_nodes, output_nodes) =
        build_image_autoencoder_network(image.width, image.height, device);
    let mut diffusion = DiffusionLoop::new(DiffusionConfig {
        alpha_schedule: AnnealingSchedule::constant(0.5),
        sigma_schedule: AnnealingSchedule::constant(0.0),
        tolerance: 1e-3,
        jt_tolerance: 5e-4,
        stability_tolerance: 5e-4,
        stability_window: 2,
        max_energy_increase: usize::MAX,
        max_iters: 20,
        entropy_policy: EntropyPolicy::default(),
        fact_recruitment: None,
    });
    let scheduler = ReasoningScheduler::new(SchedulerConfig { settle_steps: 3 });
    let mut profiler = NetworkProfiler::new(ProfilerConfig {
        activation_threshold: 0.2,
    });

    let embedding = encoder.encode(image)?;
    profiler.reset();
    let outcome = scheduler.run_case(
        &mut network,
        &embedding.flatten_all()?,
        &mut diffusion,
        Some(&mut profiler),
    );

    let output_activations: Vec<f32> = output_nodes
        .iter()
        .map(|&node_id| outcome.state[node_id])
        .collect();
    let output_tensor =
        Tensor::from_vec(output_activations, (image.pixels.len(), 6), device)?;
    let decoded_image = decoder.decode(&output_tensor)?;

    Ok(decoded_image)
}
