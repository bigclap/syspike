//! Raw pixel decoder for the diffusion stack.

use candle_core::{IndexOp, Tensor};
use candle_nn::{conv_transpose2d, ConvTranspose2dConfig, Module, VarBuilder};
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
pub struct CnnPixelDecoder {
    deconv1: candle_nn::ConvTranspose2d,
    deconv2: candle_nn::ConvTranspose2d,
}

impl CnnPixelDecoder {
    pub fn new(vs: &mut VarBuilder) -> Result<Self, DecoderError> {
        let deconv1 = conv_transpose2d(64, 32, 5, ConvTranspose2dConfig::default(), vs.pp("dc1"))?;
        let deconv2 = conv_transpose2d(32, 1, 5, ConvTranspose2dConfig::default(), vs.pp("dc2"))?;
        Ok(Self { deconv1, deconv2 })
    }

    /// Decodes a single tensor of shape `(seq_len, 64)` into an image.
    pub fn decode(&self, tensor: &Tensor) -> Result<Image, DecoderError> {
        let h = 4;
        let w = 4;
        let tensor = tensor.reshape((1, 64, h, w))?;
        let tensor = self.deconv1.forward(&tensor)?;
        let tensor = tensor.upsample_nearest2d(14, 14)?;
        let tensor = self.deconv2.forward(&tensor)?;
        let tensor = tensor.upsample_nearest2d(28, 28)?;
        let (_b, c, h, w) = tensor.dims4()?;
        let tensor = tensor.permute((0, 2, 3, 1))?.reshape((h * w, c))?;

        let pixel_data = tensor.to_vec2::<f32>()?;
        let pixels: Vec<Pixel> = pixel_data
            .iter()
            .map(|p| {
                let val = (p[0] * 255.0) as u8;
                Pixel {
                    r: val,
                    g: val,
                    b: val,
                    a: 255,
                }
            })
            .collect();

        Ok(Image {
            pixels,
            width: 28,
            height: 28,
        })
    }

    /// Decodes a batch of tensors of shape `(batch, seq_len, 64)` into images.
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
