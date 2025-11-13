use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::Device;
use image::{ImageBuffer, Rgba};
use trainer::mnist_autoencoder::{Autoencoder, MnistAutoencoderTrainer};

use crate::config::{TrainSettings, InferSettings};
use crate::telemetry::{write_profile, PROFILER};

pub fn run_train(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<TrainSettings>("train", config_path)?;

    let trainer_config = settings.trainer;
    let trainer = MnistAutoencoderTrainer::new(trainer_config);
    trainer.train()?;

    if let Some(profile_path) = settings.profile_output {
        if let Some(guard) = PROFILER.lock().unwrap().take() {
            ensure_parent(&profile_path)?;
            write_profile(guard, &profile_path);
            println!("CPU profile written to {}", profile_path.display());
        }
    }

    Ok(())
}

pub fn run_eval(_config_path: Option<PathBuf>) -> Result<()> {
    // TODO: Implement evaluation for the MNIST autoencoder.
    Ok(())
}

pub fn run_infer(config_path: Option<PathBuf>) -> Result<()> {
    let _settings = crate::config::load_settings::<InferSettings>("infer", config_path)?;
    let device = Device::Cpu;
    let dataset = datasets::load_mnist()?;
    let image = &dataset.images[0];

    let mut autoencoder = Autoencoder::new(device.clone())?;
    autoencoder.load("autoencoder.safetensors", &device)?;

    let encoded = autoencoder.encoder.encode(image)?;
    let decoded = autoencoder.decoder.decode(&encoded)?;

    let mut out_buffer = ImageBuffer::new(decoded.width as u32, decoded.height as u32);
    for (i, pixel) in decoded.pixels.iter().enumerate() {
        let x = i as u32 % decoded.width as u32;
        let y = i as u32 / decoded.width as u32;
        out_buffer.put_pixel(x, y, Rgba([pixel.r, pixel.g, pixel.b, pixel.a]));
    }

    let output_path = "output.png";
    out_buffer.save(output_path)?;
    println!("Output image saved to {}", output_path);

    Ok(())
}

pub fn run_profile(_config_path: Option<PathBuf>) -> Result<()> {
    // TODO: Implement profiling for the MNIST autoencoder.
    Ok(())
}

fn write_text_file(path: &Path, contents: &str) -> Result<()> {
    ensure_parent(path)?;
    let mut body = contents.to_string();
    if !body.ends_with('\n') {
        body.push('\n');
    }
    fs::write(path, body).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }
    Ok(())
}
