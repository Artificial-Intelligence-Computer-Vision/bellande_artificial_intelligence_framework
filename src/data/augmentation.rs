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

use rand::Rng;

pub trait Transform: Send + Sync {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError>;
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Compose { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        let mut current = tensor.clone();
        for transform in &self.transforms {
            current = transform.apply(&current)?;
        }
        Ok(current)
    }
}

pub struct RandomHorizontalFlip {
    p: f32,
}

impl RandomHorizontalFlip {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p <= 1.0);
        RandomHorizontalFlip { p }
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        if tensor.shape.len() != 4 {
            return Err(BellandeError::InvalidShape(format!(
                "Transform Invalid Shape"
            )));
        }

        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() > self.p {
            return Ok(tensor.clone());
        }

        let (batch_size, channels, height, width) = (
            tensor.shape[0],
            tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
        );

        let mut flipped_data = vec![0.0; tensor.data.len()];
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let src_idx = ((b * channels + c) * height + h) * width + w;
                        let dst_idx = ((b * channels + c) * height + h) * width + (width - 1 - w);
                        flipped_data[dst_idx] = tensor.data[src_idx];
                    }
                }
            }
        }

        Ok(Tensor::new(
            flipped_data,
            tensor.shape.clone(),
            tensor.requires_grad,
            tensor.device.clone(),
            tensor.dtype,
        ))
    }
}

pub struct RandomRotation {
    degrees: (f32, f32),
}

impl RandomRotation {
    pub fn new(degrees: (f32, f32)) -> Self {
        RandomRotation { degrees }
    }
}

impl Transform for RandomRotation {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor, BellandeError> {
        // Implementation for random rotation
        unimplemented!()
    }
}
