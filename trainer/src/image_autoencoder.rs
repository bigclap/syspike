use anyhow::Result;
use candle_core::{Device, Tensor};
use core_graph::NetworkProfiler;
use core_rules::{
    diffusion::{AnnealingSchedule, DiffusionConfig, DiffusionLoop, EntropyPolicy},
    scheduler::{ReasoningScheduler, SchedulerConfig},
};
use datasets::ImageDataset;
use evalbench::autoencoder::build_image_autoencoder_network;
use image::ImageError;
use model_image_enc::Image;
use std::path::Path;
use thiserror::Error;
use anyhow::bail;

#[derive(Debug, Error)]
pub enum TrainerError {
    #[error("image error: {0}")]
    Image(#[from] ImageError),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("dataset error: {0}")]
    Dataset(#[from] datasets::DatasetError),
    #[error("inconsistent image dimensions in dataset")]
    InconsistentDimensions,
}

pub struct ValidationRecord {
    pub input_image: Image,
    pub output_image: Image,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    pub mse: f32,
}

pub struct ImageAutoencoderTrainer {}

impl ImageAutoencoderTrainer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn validate(&self, dataset_path: &Path) -> Result<ValidationReport> {
        let device = Device::Cpu;
        let dataset = ImageDataset::from_directory(dataset_path)?;

        if dataset.images.is_empty() {
            bail!("dataset is empty");
        }

        let first_image = &dataset.images[0];
        let (width, height) = (first_image.width, first_image.height);

        for image in &dataset.images {
            if image.width != width || image.height != height {
                return Err(TrainerError::InconsistentDimensions.into());
            }
        }

        let (mut network, encoder, decoder, _input_nodes, output_nodes) =
            build_image_autoencoder_network(width, height, &device);
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
        let mut profiler = NetworkProfiler::new(core_graph::ProfilerConfig {
            activation_threshold: 0.2,
        });

        let mut records = Vec::new();

        for image in &dataset.images {
            let embedding = encoder.encode(&image)?;
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
                Tensor::from_vec(output_activations, (image.pixels.len(), 6), &device)?;
            let decoded_image = decoder.decode(&output_tensor)?;
            records.push(ValidationRecord {
                input_image: image.clone(),
                output_image: decoded_image,
            });
        }

        let mut mse_sum = 0.0;
        for record in records {
            mse_sum += calculate_mse(&record.input_image, &record.output_image);
        }
        let mse = mse_sum / dataset.images.len() as f32;

        Ok(ValidationReport { mse })
    }
}

fn calculate_mse(image1: &Image, image2: &Image) -> f32 {
    let mut mse = 0.0;
    for (p1, p2) in image1.pixels.iter().zip(image2.pixels.iter()) {
        mse += (p1.r as f32 - p2.r as f32).powi(2);
        mse += (p1.g as f32 - p2.g as f32).powi(2);
        mse += (p1.b as f32 - p2.b as f32).powi(2);
        mse += (p1.a as f32 - p2.a as f32).powi(2);
    }
    mse / (image1.pixels.len() * 4) as f32
}
