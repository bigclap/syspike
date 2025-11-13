use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, loss, ParamsAdamW, VarMap};
use datasets::load_mnist;
use model_image_dec::CnnPixelDecoder;
use model_image_enc::CnnImageEncoder;
use crate::config::OfflineTrainerConfig;
use std::path::Path;

pub struct Autoencoder {
    encoder: CnnImageEncoder,
    decoder: CnnPixelDecoder,
    varmap: VarMap,
}

impl Autoencoder {
    pub fn new(device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let mut vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let encoder = CnnImageEncoder::new(&mut vb.pp("encoder"), device)?;
        let decoder = CnnPixelDecoder::new(&mut vb.pp("decoder"))?;
        Ok(Self { encoder, decoder, varmap })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(&mut self, path: P, device: &Device) -> Result<()> {
        self.varmap.load(path)?;
        let mut vb = VarBuilder::from_varmap(&self.varmap, candle_core::DType::F32, device);
        self.encoder = CnnImageEncoder::new(&mut vb.pp("encoder"), device.clone())?;
        self.decoder = CnnPixelDecoder::new(&mut vb.pp("decoder"))?;
        Ok(())
    }
}

pub struct MnistAutoencoderTrainer {
    config: OfflineTrainerConfig,
}

impl MnistAutoencoderTrainer {
    pub fn new(config: OfflineTrainerConfig) -> Self {
        Self { config }
    }

    pub fn train(&self) -> Result<()> {
        let device = Device::Cpu;
        let dataset = load_mnist()?;
        let mut autoencoder = Autoencoder::new(device.clone())?;

        let mut adamw_params = ParamsAdamW::default();
        adamw_params.lr = self.config.learning_rate as f64;
        adamw_params.weight_decay = self.config.weight_decay as f64;
        let vars = autoencoder.varmap.all_vars();
        let mut opt = AdamW::new(vars, adamw_params)?;

        for epoch in 0..self.config.total_steps {
            let mut total_loss = 0.0;
            for image in &dataset.images {
                let input_tensor = autoencoder.encoder.encode(image)?;
                let output_image = autoencoder.decoder.decode(&input_tensor)?;
                let pixel_data: Vec<f32> = image.pixels.iter().map(|p| p.r as f32 / 255.0).collect();
                let original_tensor = Tensor::from_vec(pixel_data, (1, 1, 28, 28), &device)?;

                let output_pixel_data: Vec<f32> = output_image.pixels.iter().map(|p| p.r as f32 / 255.0).collect();
                let output_tensor = Tensor::from_vec(output_pixel_data, (1, 1, 28, 28), &device)?;
                let loss = loss::mse(&original_tensor, &output_tensor)?;
                opt.backward_step(&loss)?;
                total_loss += loss.to_scalar::<f32>()?;
            }
            println!("Epoch: {}, Loss: {}", epoch, total_loss / dataset.images.len() as f32);
        }

        autoencoder.save("autoencoder.safetensors")?;

        Ok(())
    }
}
