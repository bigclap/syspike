//! Candle-backed image encoder utilities used by the diffusion stack.

use candle_core::{Device, Tensor};
use candle_nn::{conv2d, Conv2dConfig, Module, VarBuilder};
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
}

#[derive(Clone)]
pub struct Image {
    pub pixels: Vec<Pixel>,
    pub width: u16,
    pub height: u16,
}

/// A CNN-based encoder for images.
pub struct CnnImageEncoder {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    device: Device,
}

impl CnnImageEncoder {
    pub fn new(vs: &mut VarBuilder, device: Device) -> Result<Self, EncoderError> {
        let conv1 = conv2d(1, 32, 5, Conv2dConfig::default(), vs.pp("c1"))?;
        let conv2 = conv2d(32, 64, 5, Conv2dConfig::default(), vs.pp("c2"))?;
        Ok(Self { conv1, conv2, device })
    }

    /// Encodes a single image, returning a tensor of shape `(new_width * new_height, 64)`.
    pub fn encode(&self, image: &Image) -> Result<Tensor, EncoderError> {
        let pixel_data: Vec<f32> = image
            .pixels
            .iter()
            .map(|p| p.r as f32 / 255.0)
            .collect();

        let tensor = Tensor::from_vec(
            pixel_data,
            (1, 1, image.height as usize, image.width as usize),
            &self.device,
        )?;
        let tensor = self.conv1.forward(&tensor)?;
        let tensor = tensor.max_pool2d(2)?;
        let tensor = self.conv2.forward(&tensor)?;
        let tensor = tensor.max_pool2d(2)?;
        let (b, c, h, w) = tensor.dims4()?;
        tensor.reshape((b, c, h * w))?.squeeze(0).map_err(EncoderError::Candle)
    }

    /// Encodes a batch of images returning a tensor with shape `(batch, new_width * new_height, 64)`.
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
