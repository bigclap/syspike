//! Candle-backed image encoder utilities used by the diffusion stack.

use candle_core::{Device, Tensor};
use thiserror::Error;

/// Errors that may occur while loading or running the encoder.
#[derive(Debug, Error)]
pub enum EncoderError {
    /// Raised when candle operations fail.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Raised when configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
    pub x: u16,
    pub y: u16,
}

#[derive(Clone)]
pub struct Image {
    pub pixels: Vec<Pixel>,
    pub width: u16,
    pub height: u16,
}

/// A frozen encoder for images.
pub struct FrozenImageEncoder {
    device: Device,
    width: u16,
    height: u16,
}

impl FrozenImageEncoder {
    pub fn new(width: u16, height: u16, device: Device) -> Result<Self, EncoderError> {
        Ok(Self {
            device,
            width,
            height,
        })
    }

    /// Encodes a single image, returning a tensor of shape `(width * height, 6)`.
    pub fn encode(&self, image: &Image) -> Result<Tensor, EncoderError> {
        if image.width != self.width || image.height != self.height {
            return Err(EncoderError::InvalidConfig(format!(
                "image dimensions ({}, {}) do not match encoder dimensions ({}, {})",
                image.width, image.height, self.width, self.height
            )));
        }

        let pixel_data: Vec<f32> = image
            .pixels
            .iter()
            .flat_map(|p| {
                [
                    p.r as f32 / 255.0,
                    p.g as f32 / 255.0,
                    p.b as f32 / 255.0,
                    p.a as f32 / 255.0,
                    p.x as f32 / self.width as f32,
                    p.y as f32 / self.height as f32,
                ]
            })
            .collect();

        let tensor = Tensor::from_vec(pixel_data, (image.pixels.len(), 6), &self.device)?;
        Ok(tensor)
    }

    /// Encodes a batch of images returning a tensor with shape `(batch, width * height, 6)`.
    pub fn encode_batch(&self, images: &[Image]) -> Result<Tensor, EncoderError> {
        let mut batch_tensors = Vec::new();
        for image in images {
            let tensor = self.encode(image)?;
            batch_tensors.push(tensor);
        }
        let batch_tensor = Tensor::stack(&batch_tensors, 0)?;
        Ok(batch_tensor)
    }
}
