use evalbench::autoencoder::run_image_autoencoder_inference;
use image::{ImageBuffer, Rgba};
use tempfile::tempdir;
use candle_core::Device;
use datasets::load_image;

#[test]
fn test_image_autoencoder_inference_runs_successfully() {
    let dir = tempdir().unwrap();
    let image_path = dir.path().join("test.png");

    // Create a dummy image
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(10, 10);
    img.save(&image_path).unwrap();

    let image = load_image(&image_path).unwrap();

    // Run the inference
    let output_image = run_image_autoencoder_inference(&image, &Device::Cpu).unwrap();

    assert_eq!(image.width, output_image.width);
    assert_eq!(image.height, output_image.height);
}
