//! Raw pixel decoder for the diffusion stack.

use candle_core::{IndexOp, Tensor};
use model_image_enc::{Image, Pixel};
use thiserror::Error;

/// Errors that may occur while decoding.
#[derive(Debug, Error)]
pub enum DecoderError {
    /// Raised when candle operations fail.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Raised when configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// A decoder for raw pixel data.
pub struct RawPixelDecoder {
    width: u16,
    height: u16,
}

impl RawPixelDecoder {
    pub fn new(width: u16, height: u16) -> Self {
        Self { width, height }
    }

    /// Decodes a single tensor of shape `(width * height, 6)` into an image.
    pub fn decode(&self, tensor: &Tensor) -> Result<Image, DecoderError> {
        let expected_len = self.width as usize * self.height as usize;
        if tensor.dim(0)? != expected_len || tensor.dim(1)? != 6 {
            return Err(DecoderError::InvalidConfig(format!(
                "tensor dimensions ({:?}) do not match expected dimensions ({}, 6)",
                tensor.dims(),
                expected_len
            )));
        }

        let pixel_data = tensor.to_vec2::<f32>()?;
        let pixels: Vec<Pixel> = pixel_data
            .iter()
            .map(|p| Pixel {
                r: (p[0] * 255.0) as u8,
                g: (p[1] * 255.0) as u8,
                b: (p[2] * 255.0) as u8,
                a: (p[3] * 255.0) as u8,
                x: (p[4] * self.width as f32) as u16,
                y: (p[5] * self.height as f32) as u16,
            })
            .collect();

        Ok(Image {
            pixels,
            width: self.width,
            height: self.height,
        })
    }

    /// Decodes a batch of tensors of shape `(batch, width * height, 6)` into images.
    pub fn decode_batch(&self, batch_tensor: &Tensor) -> Result<Vec<Image>, DecoderError> {
        let batch_size = batch_tensor.dim(0)?;
        let mut images = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let tensor = batch_tensor.i(i)?;
            let image = self.decode(&tensor)?;
            images.push(image);
        }
        Ok(images)
    }
}
