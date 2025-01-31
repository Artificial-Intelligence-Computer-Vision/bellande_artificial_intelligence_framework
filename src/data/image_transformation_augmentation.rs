// Copyright (C) 2024 Bellande Artificial Intelligence Computer Vision Research Innovation Center, Ronaldson Bellande

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use crate::core::{error::BellandeError, tensor::Tensor};
use rand::{thread_rng, Rng};

/// Trait for image transformations
pub trait Transform: Send + Sync {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError>;
    fn name(&self) -> &str;
}

/// Center crop transformation
pub struct CenterCrop {
    height: usize,
    width: usize,
}

impl CenterCrop {
    pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }
}

impl Transform for CenterCrop {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        let shape = tensor.shape();
        if shape.len() != 4 {
            return Err(BellandeError::InvalidShape(
                "Expected 4D tensor".to_string(),
            ));
        }

        let [batch_size, channels, in_height, in_width] = shape[..4] else {
            return Err(BellandeError::InvalidShape(
                "Invalid tensor shape".to_string(),
            ));
        };

        if in_height < self.height || in_width < self.width {
            return Err(BellandeError::InvalidOperation(
                "Crop size larger than input size".into(),
            ));
        }

        let start_h = (in_height - self.height) / 2;
        let start_w = (in_width - self.width) / 2;
        let mut cropped = vec![0.0; batch_size * channels * self.height * self.width];

        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..self.height {
                    for w in 0..self.width {
                        let src_idx = ((b * channels + c) * in_height + (start_h + h)) * in_width
                            + (start_w + w);
                        let dst_idx = ((b * channels + c) * self.height + h) * self.width + w;
                        cropped[dst_idx] = tensor.data()[src_idx];
                    }
                }
            }
        }

        Ok(Tensor::new(
            cropped,
            vec![batch_size, channels, self.height, self.width],
            tensor.requires_grad,
            tensor.device.clone(),
            tensor.dtype,
        ))
    }

    fn name(&self) -> &str {
        "CenterCrop"
    }
}

/// Random crop transformation
pub struct RandomCrop {
    height: usize,
    width: usize,
}

impl RandomCrop {
    pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }
}

impl Transform for RandomCrop {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        let shape = tensor.shape();
        if shape.len() != 4 {
            return Err(BellandeError::InvalidShape(
                "Expected 4D tensor".to_string(),
            ));
        }

        let [batch_size, channels, in_height, in_width] = shape[..4] else {
            return Err(BellandeError::InvalidShape(
                "Invalid tensor shape".to_string(),
            ));
        };

        if in_height < self.height || in_width < self.width {
            return Err(BellandeError::InvalidOperation(
                "Crop size larger than input size".into(),
            ));
        }

        let mut rng = thread_rng();
        let start_h = rng.gen_range(0..=in_height - self.height);
        let start_w = rng.gen_range(0..=in_width - self.width);
        let mut cropped = vec![0.0; batch_size * channels * self.height * self.width];

        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..self.height {
                    for w in 0..self.width {
                        let src_idx = ((b * channels + c) * in_height + (start_h + h)) * in_width
                            + (start_w + w);
                        let dst_idx = ((b * channels + c) * self.height + h) * self.width + w;
                        cropped[dst_idx] = tensor.data()[src_idx];
                    }
                }
            }
        }

        Ok(Tensor::new(
            cropped,
            vec![batch_size, channels, self.height, self.width],
            tensor.requires_grad,
            tensor.device.clone(),
            tensor.dtype,
        ))
    }

    fn name(&self) -> &str {
        "RandomCrop"
    }
}

pub struct RandomVerticalFlip {
    probability: f32,
}

impl RandomVerticalFlip {
    pub fn new(probability: f32) -> Self {
        Self { probability }
    }
}

impl Transform for RandomVerticalFlip {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        if thread_rng().gen::<f32>() > self.probability {
            return Ok(tensor.clone());
        }

        let shape = tensor.shape();
        if shape.len() != 4 {
            return Err(BellandeError::InvalidShape("Expected 4D tensor".into()));
        }

        let [batch_size, channels, height, width] = shape[..4] else {
            return Err(BellandeError::InvalidShape("Invalid tensor shape".into()));
        };

        let mut flipped = vec![0.0; tensor.data.len()];
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let src_idx = ((b * channels + c) * height + h) * width + w;
                        let dst_idx = ((b * channels + c) * height + (height - 1 - h)) * width + w;
                        flipped[dst_idx] = tensor.data[src_idx];
                    }
                }
            }
        }

        Ok(Tensor::new(
            flipped,
            shape.to_vec(),
            tensor.requires_grad,
            tensor.device.clone(),
            tensor.dtype,
        ))
    }

    fn name(&self) -> &str {
        "RandomVerticalFlip"
    }
}

pub struct ColorJitter {
    brightness: f32,
    contrast: f32,
    saturation: f32,
}

impl ColorJitter {
    pub fn new(brightness: f32, contrast: f32, saturation: f32) -> Self {
        Self {
            brightness,
            contrast,
            saturation,
        }
    }

    fn adjust_brightness(&self, data: &mut [f32]) {
        let factor = 1.0 + thread_rng().gen_range(-self.brightness..=self.brightness);
        for value in data.iter_mut() {
            *value = (*value * factor).max(0.0).min(1.0);
        }
    }

    fn adjust_contrast(&self, data: &mut [f32]) {
        let factor = 1.0 + thread_rng().gen_range(-self.contrast..=self.contrast);
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        for value in data.iter_mut() {
            *value = ((*value - mean) * factor + mean).max(0.0).min(1.0);
        }
    }

    fn adjust_saturation(&self, data: &mut [f32], shape: &[usize]) {
        if shape[1] != 3 {
            return;
        }

        let factor = 1.0 + thread_rng().gen_range(-self.saturation..=self.saturation);
        let size = shape[0] * shape[2] * shape[3];

        for i in 0..size {
            let r = data[i];
            let g = data[i + size];
            let b = data[i + size * 2];
            let gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;

            data[i] = ((r - gray) * factor + gray).max(0.0).min(1.0);
            data[i + size] = ((g - gray) * factor + gray).max(0.0).min(1.0);
            data[i + size * 2] = ((b - gray) * factor + gray).max(0.0).min(1.0);
        }
    }
}

impl Transform for ColorJitter {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        let shape = tensor.shape().to_vec();
        let mut data = tensor.data().to_vec();

        self.adjust_brightness(&mut data);
        self.adjust_contrast(&mut data);
        self.adjust_saturation(&mut data, &shape);

        Ok(Tensor::new(
            data,
            shape,
            tensor.requires_grad,
            tensor.device.clone(),
            tensor.dtype,
        ))
    }

    fn name(&self) -> &str {
        "ColorJitter"
    }
}

/// Gaussian noise transformation
pub struct GaussianNoise {
    mean: f32,
    std: f32,
}

impl GaussianNoise {
    pub fn new(mean: f32, std: f32) -> Self {
        Self { mean, std }
    }
}

impl Transform for GaussianNoise {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        let mut rng = thread_rng();
        let mut noisy = tensor.data.to_vec();
        let shape = tensor.shape().to_vec();

        for value in noisy.iter_mut() {
            let noise = rng.gen_range(-2.0..=2.0) * self.std + self.mean;
            *value = (*value + noise).max(0.0).min(1.0);
        }

        Ok(Tensor::new(
            noisy,
            shape,
            tensor.requires_grad,
            tensor.device.clone(),
            tensor.dtype,
        ))
    }

    fn name(&self) -> &str {
        "GaussianNoise"
    }
}
