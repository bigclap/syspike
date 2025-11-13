//! Dataset loader for MNIST.

use candle_core::Tensor;
use candle_datasets::vision::mnist;
use thiserror::Error;
use model_image_enc::{Image, Pixel};

/// Errors that may occur while loading a dataset.
#[derive(Debug, Error)]
pub enum DatasetError {
    /// Raised when candle operations fail.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Raised when a path is invalid.
    #[error("invalid path: {0}")]
    InvalidPath(String),
    /// Raised when IO operations fail.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A dataset of MNIST images.
pub struct MnistDataset {
    pub images: Vec<Image>,
    pub labels: Tensor,
}

pub fn load_mnist() -> Result<MnistDataset, DatasetError> {
    let m = mnist::load()?;
    let images_tensor = m.train_images.to_dtype(candle_core::DType::F32)?;
    let labels = m.train_labels;
    let (num_images, _num_pixels) = images_tensor.dims2()?;
    let images_vec = images_tensor.to_vec2::<f32>()?;

    let mut images = Vec::with_capacity(num_images);
    for i in 0..num_images {
        let pixels: Vec<Pixel> = images_vec[i]
            .iter()
            .map(|p| {
                let val = (*p * 255.0) as u8;
                Pixel {
                    r: val,
                    g: val,
                    b: val,
                    a: 255,
                }
            })
            .collect();
        images.push(Image {
            pixels,
            width: 28,
            height: 28,
        });
    }

    Ok(MnistDataset { images, labels })
}
