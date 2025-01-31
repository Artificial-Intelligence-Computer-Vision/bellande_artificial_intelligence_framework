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

pub trait Preprocessor: Send + Sync {
    fn process(&self, tensor: &Tensor) -> Result<Tensor, BellandeError>;
}

pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(mean.len(), std.len());
        Normalize { mean, std }
    }
}

impl Preprocessor for Normalize {
    fn process(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        if tensor.shape.len() != 4 {
            return Err(BellandeError::InvalidShape(format!("Preprocessor Invalid")))?;
        }

        let (batch_size, channels, height, width) = (
            tensor.shape[0],
            tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
        );

        assert_eq!(channels, self.mean.len());

        let mut normalized = tensor.data.clone();
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        normalized[idx] = (normalized[idx] - self.mean[c]) / self.std[c];
                    }
                }
            }
        }

        Ok(Tensor::new(
            normalized,
            tensor.shape.clone(),
            tensor.requires_grad,
            tensor.device.clone(),
            tensor.dtype,
        ))
    }
}
