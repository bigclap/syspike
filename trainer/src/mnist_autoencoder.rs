use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, loss, ParamsAdamW, VarMap};
use datasets::load_mnist;
use model_image_dec::CnnPixelDecoder;
use model_image_enc::CnnImageEncoder;
use crate::config::OfflineTrainerConfig;

pub struct Autoencoder {
    pub encoder: CnnImageEncoder,
    pub decoder: CnnPixelDecoder,
}

impl Autoencoder {
    pub fn new(vs: &mut VarBuilder, device: Device) -> Result<Self> {
        let encoder = CnnImageEncoder::new(&mut vs.pp("encoder"), device)?;
        let decoder = CnnPixelDecoder::new(&mut vs.pp("decoder"))?;
        Ok(Self { encoder, decoder })
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
        let varmap = VarMap::new();
        let mut vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let autoencoder = Autoencoder::new(&mut vb, device.clone())?;

        let mut adamw_params = ParamsAdamW::default();
        adamw_params.lr = self.config.learning_rate as f64;
        adamw_params.weight_decay = self.config.weight_decay as f64;
        let vars = varmap.all_vars();
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

        Ok(())
    }
}
