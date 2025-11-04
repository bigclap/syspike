use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::Device;
use evalbench::autoencoder::run_image_autoencoder_inference;
use image::{ImageBuffer, Rgba};
use tracing::info;
use trainer::ImageAutoencoderTrainer;

use crate::config::{EvalSettings, InferSettings, ProfileSettings, TrainSettings};
use crate::telemetry::{init_telemetry, write_profile};

pub fn run_train(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<TrainSettings>("train", config_path)?;
    let profiler_guard = init_telemetry();

    if let Some(dataset_path) = &settings.dataset {
        info!(path = %dataset_path.display(), "training dataset located");
        let trainer = ImageAutoencoderTrainer::new();
        let report = trainer.validate(dataset_path)?;
        let report_text = format!("{:?}", report);
        println!("Training summary:\n{report_text}");

        let checkpoint_body = format!("# Offline decoder checkpoint summary\n{report_text}");
        write_text_file(&settings.checkpoint, &checkpoint_body)?;
        println!(
            "Checkpoint summary written to {}",
            settings.checkpoint.display()
        );
    } else {
        info!("no dataset supplied, training skipped");
    }

    if let Some(profile_path) = settings.profile_output {
        if let Some(guard) = profiler_guard {
            ensure_parent(&profile_path)?;
            write_profile(guard, &profile_path);
            println!("CPU profile written to {}", profile_path.display());
        }
    }

    Ok(())
}

pub fn run_eval(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<EvalSettings>("eval", config_path)?;
    let profiler_guard = init_telemetry();

    if let Some(dataset_path) = &settings.dataset {
        info!(path = %dataset_path.display(), "evaluating dataset");
        let trainer = ImageAutoencoderTrainer::new();
        let report = trainer.validate(dataset_path)?;
        let report_text = format!("{:?}", report);
        println!("Evaluation summary:\n{report_text}");

        let report_body = format!("# Evaluation summary\n{report_text}");
        write_text_file(&settings.report, &report_body)?;
        println!("Evaluation report written to {}", settings.report.display());
    } else {
        info!("no dataset supplied, evaluation skipped");
    }

    if let Some(profile_path) = settings.profile_output {
        if let Some(guard) = profiler_guard {
            ensure_parent(&profile_path)?;
            write_profile(guard, &profile_path);
            println!("CPU profile written to {}", profile_path.display());
        }
    }

    Ok(())
}

pub fn run_infer(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<InferSettings>("infer", config_path)?;
    let profiler_guard = init_telemetry();

    if let Some(checkpoint) = &settings.checkpoint {
        println!("Using checkpoint hint: {}", checkpoint.display());
    }

    for (i, input_path) in settings.inputs.iter().enumerate() {
        let image = datasets::load_image(input_path)?;
        let decoded_image = run_image_autoencoder_inference(&image, &Device::Cpu)?;
        let mut out_buffer = ImageBuffer::new(decoded_image.width as u32, decoded_image.height as u32);
        for pixel in decoded_image.pixels {
            out_buffer.put_pixel(
                pixel.x as u32,
                pixel.y as u32,
                Rgba([pixel.r, pixel.g, pixel.b, pixel.a]),
            );
        }
        let output_path = input_path.with_file_name(format!("output_{}.png", i));
        out_buffer.save(output_path)?;
    }

    if let Some(profile_path) = settings.profile_output {
        if let Some(guard) = profiler_guard {
            ensure_parent(&profile_path)?;
            write_profile(guard, &profile_path);
            println!("CPU profile written to {}", profile_path.display());
        }
    }

    Ok(())
}

pub fn run_profile(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<ProfileSettings>("profile", config_path)?;
    let profiler_guard = init_telemetry();

    for (i, input_path) in settings.inputs.iter().enumerate() {
        let image = datasets::load_image(input_path)?;
        let decoded_image = run_image_autoencoder_inference(&image, &Device::Cpu)?;
        let mut out_buffer = ImageBuffer::new(decoded_image.width as u32, decoded_image.height as u32);
        for pixel in decoded_image.pixels {
            out_buffer.put_pixel(
                pixel.x as u32,
                pixel.y as u32,
                Rgba([pixel.r, pixel.g, pixel.b, pixel.a]),
            );
        }
        let output_path = input_path.with_file_name(format!("output_{}.png", i));
        out_buffer.save(output_path)?;
    }

    if let Some(guard) = profiler_guard {
        ensure_parent(&settings.profile_output)?;
        write_profile(guard, &settings.profile_output);
        println!(
            "CPU profile written to {}",
            settings.profile_output.display()
        );
    }

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
