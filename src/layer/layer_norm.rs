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

pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f32,
    input_cache: Option<LayerNormCache>,
}

struct LayerNormCache {
    input: Tensor,
    normalized: Tensor,
    std: Vec<f32>,
    mean: Vec<f32>,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32, elementwise_affine: bool) -> Self {
        let weight = if elementwise_affine {
            Some(Tensor::ones(&normalized_shape))
        } else {
            None
        };

        let bias = if elementwise_affine {
            Some(Tensor::zeros(&normalized_shape))
        } else {
            None
        };

        LayerNorm {
            normalized_shape,
            weight,
            bias,
            eps,
            input_cache: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let batch_size = input.shape[0];
        let feature_size: usize = self.normalized_shape.iter().product();

        let input_shape: Vec<usize> = input.shape[1..].to_vec();
        let norm_shape: Vec<usize> = self.normalized_shape.clone();

        if input_shape != norm_shape {
            return Err(BellandeError::InvalidShape(format!(
                "Expected shape {:?}, got {:?}",
                norm_shape, input_shape
            )));
        }
        let mut output = input.data.clone();
        let mut mean = vec![0.0; batch_size];
        let mut std = vec![0.0; batch_size];

        // Calculate mean and standard deviation
        for b in 0..batch_size {
            let start_idx = b * feature_size;
            let end_idx = start_idx + feature_size;
            let batch_data = &input.data[start_idx..end_idx];

            // Calculate mean
            mean[b] = batch_data.iter().sum::<f32>() / feature_size as f32;

            // Calculate variance
            let variance: f32 = batch_data
                .iter()
                .map(|&x| (x - mean[b]).powi(2))
                .sum::<f32>()
                / feature_size as f32;

            std[b] = (variance + self.eps).sqrt();

            // Normalize
            for i in 0..feature_size {
                let idx = start_idx + i;
                output[idx] = (input.data[idx] - mean[b]) / std[b];

                // Apply affine transform if available
                if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                    output[idx] = output[idx] * weight.data[i] + bias.data[i];
                }
            }
        }

        // Cache for backward pass
        self.input_cache = Some(LayerNormCache {
            input: input.clone(),
            normalized: Tensor::new(
                output.clone(),
                input.shape.clone(),
                true,
                input.device.clone(),
                input.dtype,
            ),
            std,
            mean,
        });

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }

    pub fn backward(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        if let Some(ref cache) = self.input_cache {
            let batch_size = grad_output.shape[0];
            let feature_size = self.normalized_shape.iter().product();
            let mut grad_input = vec![0.0; grad_output.data.len()];

            for b in 0..batch_size {
                let start_idx = b * feature_size;
                let end_idx = start_idx + feature_size;

                let batch_grad = &grad_output.data[start_idx..end_idx];
                let batch_input = &cache.input.data[start_idx..end_idx];
                let mean = cache.mean[b];
                let std = cache.std[b];

                // Calculate gradients
                let mut sum_grad = 0.0;
                let mut sum_grad_h = 0.0;

                for i in 0..feature_size {
                    let idx = start_idx + i;
                    let h = (batch_input[i] - mean) / std;

                    if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                        sum_grad += grad_output.data[idx] * weight.data[i];
                        sum_grad_h += grad_output.data[idx] * weight.data[i] * h;
                    } else {
                        sum_grad += grad_output.data[idx];
                        sum_grad_h += grad_output.data[idx] * h;
                    }
                }

                // Apply gradients
                for i in 0..feature_size {
                    let idx = start_idx + i;
                    let h = (batch_input[i] - mean) / std;

                    grad_input[idx] = (1.0 / (feature_size as f32 * std))
                        * (feature_size as f32 * grad_output.data[idx] - sum_grad - h * sum_grad_h);

                    if let (Some(ref weight), _) = (&self.weight, &self.bias) {
                        grad_input[idx] *= weight.data[i];
                    }
                }
            }

            Ok(Tensor::new(
                grad_input,
                grad_output.shape.clone(),
                true,
                grad_output.device.clone(),
                grad_output.dtype,
            ))
        } else {
            Err(BellandeError::RuntimeError(
                "Forward pass not called".into(),
            ))
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight.clone());
        }
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}
