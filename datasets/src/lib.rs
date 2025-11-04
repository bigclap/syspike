//! Dataset loader for image files.

use std::path::Path;
use image::{GenericImageView, ImageError};
use thiserror::Error;
use model_image_enc::{Image, Pixel};

/// Errors that may occur while loading a dataset.
#[derive(Debug, Error)]
pub enum DatasetError {
    /// Raised when an image file cannot be loaded.
    #[error("failed to load image: {0}")]
    Image(#[from] ImageError),
    /// Raised when a path is invalid.
    #[error("invalid path: {0}")]
    InvalidPath(String),
}

/// A dataset of images.
pub struct ImageDataset {
    pub images: Vec<Image>,
}

impl ImageDataset {
    /// Loads all images from a directory.
    pub fn from_directory<P: AsRef<Path>>(path: P) -> Result<Self, DatasetError> {
        let mut images = Vec::new();
        let path = path.as_ref();
        if !path.is_dir() {
            return Err(DatasetError::InvalidPath(
                path.to_str().unwrap_or("").to_string(),
            ));
        }

        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                let image = load_image(&path)?;
                images.push(image);
            }
        }
        Ok(Self { images })
    }
}

/// Loads a single image from a file path.
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<Image, DatasetError> {
    let img = image::open(path)?;
    let (width, height) = img.dimensions();
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x, y);
            pixels.push(Pixel {
                r: p[0],
                g: p[1],
                b: p[2],
                a: p[3],
                x: x as u16,
                y: y as u16,
            });
        }
    }

    Ok(Image {
        pixels,
        width: width as u16,
        height: height as u16,
    })
}

#[cfg(test)]
mod tests {
    use super::{load_image, ImageDataset};
    use image::{ImageBuffer, Rgba};

    #[test]
    fn test_load_image() {
        let path = "test_image.png";
        let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(10, 10, |x, y| {
            if (x + y) % 2 == 0 {
                Rgba([0, 0, 0, 255])
            } else {
                Rgba([255, 255, 255, 255])
            }
        });
        img.save(path).unwrap();

        let loaded_image = load_image(path).unwrap();
        assert_eq!(loaded_image.width, 10);
        assert_eq!(loaded_image.height, 10);
        assert_eq!(loaded_image.pixels.len(), 100);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_image_dataset_from_directory() {
        let dir = "test_images";
        std::fs::create_dir_all(dir).unwrap();

        for i in 0..3 {
            let path = format!("{}/test_image_{}.png", dir, i);
            let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(5, 5);
            img.save(&path).unwrap();
        }

        let dataset = ImageDataset::from_directory(dir).unwrap();
        assert_eq!(dataset.images.len(), 3);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
