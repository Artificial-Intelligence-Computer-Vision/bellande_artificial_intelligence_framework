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
use crate::models::sequential::NeuralLayer;
use std::sync::Arc;

pub struct BatchNorm1d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    pub(crate) running_mean: Arc<Tensor>,
    pub(crate) running_var: Arc<Tensor>,
    pub(crate) weight: Option<Tensor>,
    pub(crate) bias: Option<Tensor>,
    pub(crate) training: bool,
    input: Option<Tensor>,
}

pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    pub(crate) running_mean: Arc<Tensor>,
    pub(crate) running_var: Arc<Tensor>,
    pub(crate) weight: Option<Tensor>,
    pub(crate) bias: Option<Tensor>,
    pub(crate) training: bool,
    input: Option<Tensor>,
}

impl BatchNorm1d {
    pub fn new(num_features: usize, eps: f32, momentum: f32, affine: bool) -> Self {
        let running_mean = Arc::new(Tensor::zeros(&[num_features]));
        let running_var = Arc::new(Tensor::ones(&[num_features]));

        BatchNorm1d {
            num_features,
            eps,
            momentum,
            running_mean,
            running_var,
            weight: if affine {
                Some(Tensor::ones(&[num_features]))
            } else {
                None
            },
            bias: if affine {
                Some(Tensor::zeros(&[num_features]))
            } else {
                None
            },
            training: true,
            input: None,
        }
    }

    fn update_running_stats(&mut self, mean: &[f32], var: &[f32]) -> Result<(), BellandeError> {
        let running_mean = Arc::get_mut(&mut self.running_mean).ok_or_else(|| {
            BellandeError::RuntimeError("Failed to get mutable reference to running mean".into())
        })?;

        let running_var = Arc::get_mut(&mut self.running_var).ok_or_else(|| {
            BellandeError::RuntimeError(
                "Failed to get mutable reference to running variance".into(),
            )
        })?;

        for i in 0..self.num_features {
            running_mean.data[i] =
                self.momentum * running_mean.data[i] + (1.0 - self.momentum) * mean[i];
            running_var.data[i] =
                self.momentum * running_var.data[i] + (1.0 - self.momentum) * var[i];
        }

        Ok(())
    }

    fn forward_impl(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        if input.shape.len() != 2 {
            return Err(BellandeError::InvalidShape(
                "Expected 2D tensor (batch_size, num_features)".into(),
            ));
        }

        let (batch_size, features) = (input.shape[0], input.shape[1]);

        if features != self.num_features {
            return Err(BellandeError::InvalidOperation(format!(
                "Expected {} features but got {}",
                self.num_features, features
            )));
        }

        let mut output = input.data.clone();

        if self.training {
            let mut mean = vec![0.0; features];
            let mut var = vec![0.0; features];

            // Calculate mean and variance
            for f in 0..features {
                let mut sum = 0.0;
                let mut sq_sum = 0.0;

                for b in 0..batch_size {
                    let idx = b * features + f;
                    let val = input.data[idx];
                    sum += val;
                    sq_sum += val * val;
                }

                mean[f] = sum / batch_size as f32;
                var[f] = sq_sum / batch_size as f32 - mean[f] * mean[f];
            }

            // Update running statistics
            self.update_running_stats(&mean, &var)?;

            // Normalize
            for f in 0..features {
                let std = (var[f] + self.eps).sqrt();
                for b in 0..batch_size {
                    let idx = b * features + f;
                    output[idx] = (output[idx] - mean[f]) / std;

                    if let Some(ref weight) = self.weight {
                        output[idx] *= weight.data[f];
                    }
                    if let Some(ref bias) = self.bias {
                        output[idx] += bias.data[f];
                    }
                }
            }
        } else {
            // Use running statistics for inference
            let running_mean = &self.running_mean;
            let running_var = &self.running_var;

            for f in 0..features {
                let std = (running_var.data[f] + self.eps).sqrt();
                for b in 0..batch_size {
                    let idx = b * features + f;
                    output[idx] = (output[idx] - running_mean.data[f]) / std;

                    if let Some(ref weight) = self.weight {
                        output[idx] *= weight.data[f];
                    }
                    if let Some(ref bias) = self.bias {
                        output[idx] += bias.data[f];
                    }
                }
            }
        }

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }

    fn backward_impl(&mut self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        let input = self.input.as_ref().ok_or(BellandeError::InvalidBackward(
            "Forward pass not called before backward".into(),
        ))?;

        let (batch_size, num_features) = (input.shape[0], input.shape[1]);
        let n = batch_size as f32;

        // Calculate mean and variance
        let mut mean = vec![0.0; num_features];
        let mut var = vec![0.0; num_features];

        for f in 0..num_features {
            let mut sum = 0.0;
            let mut sq_sum = 0.0;
            for b in 0..batch_size {
                let idx = b * num_features + f;
                let val = input.data[idx];
                sum += val;
                sq_sum += val * val;
            }
            mean[f] = sum / n;
            var[f] = sq_sum / n - mean[f] * mean[f];
        }

        // Initialize gradients
        let mut dx = vec![0.0; input.data.len()];
        let mut dweight = if self.weight.is_some() {
            vec![0.0; num_features]
        } else {
            vec![]
        };
        let mut dbias = if self.bias.is_some() {
            vec![0.0; num_features]
        } else {
            vec![]
        };

        // Compute gradients
        for f in 0..num_features {
            let std = (var[f] + self.eps).sqrt();
            let inv_std = 1.0 / std;

            let mut dxhat = vec![0.0; batch_size];
            let mut sum_dxhat = 0.0;
            let mut sum_dxhat_x = 0.0;

            // Compute dxhat and accumulate sums
            for b in 0..batch_size {
                let idx = b * num_features + f;
                let xhat = (input.data[idx] - mean[f]) * inv_std;

                dxhat[b] = grad_output.data[idx];
                if let Some(ref weight) = self.weight {
                    dxhat[b] *= weight.data[f];
                }

                sum_dxhat += dxhat[b];
                sum_dxhat_x += dxhat[b] * xhat;
            }

            // Compute dx
            for b in 0..batch_size {
                let idx = b * num_features + f;
                let xhat = (input.data[idx] - mean[f]) * inv_std;

                dx[idx] = inv_std * (dxhat[b] - sum_dxhat / n - xhat * sum_dxhat_x / n);
            }

            // Compute dweight and dbias if they exist
            if let Some(_) = self.weight {
                dweight[f] = 0.0;
                for b in 0..batch_size {
                    let idx = b * num_features + f;
                    let xhat = (input.data[idx] - mean[f]) * inv_std;
                    dweight[f] += grad_output.data[idx] * xhat;
                }
            }

            if let Some(_) = self.bias {
                dbias[f] = 0.0;
                for b in 0..batch_size {
                    let idx = b * num_features + f;
                    dbias[f] += grad_output.data[idx];
                }
            }
        }

        // Update weight and bias gradients if they exist
        if let Some(ref mut weight) = self.weight {
            weight.grad = Some(dweight);
        }

        if let Some(ref mut bias) = self.bias {
            bias.grad = Some(dbias);
        }

        Ok(Tensor::new(
            dx,
            input.shape.clone(),
            true,
            input.device.clone(),
            input.dtype,
        ))
    }
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f32, momentum: f32, affine: bool) -> Self {
        let running_mean = Arc::new(Tensor::zeros(&[num_features]));
        let running_var = Arc::new(Tensor::ones(&[num_features]));

        BatchNorm2d {
            num_features,
            eps,
            momentum,
            running_mean,
            running_var,
            weight: if affine {
                Some(Tensor::ones(&[num_features]))
            } else {
                None
            },
            bias: if affine {
                Some(Tensor::zeros(&[num_features]))
            } else {
                None
            },
            training: true,
            input: None,
        }
    }

    fn update_running_stats(&mut self, mean: &[f32], var: &[f32]) -> Result<(), BellandeError> {
        let running_mean = Arc::get_mut(&mut self.running_mean).ok_or_else(|| {
            BellandeError::RuntimeError("Failed to get mutable reference to running mean".into())
        })?;

        let running_var = Arc::get_mut(&mut self.running_var).ok_or_else(|| {
            BellandeError::RuntimeError(
                "Failed to get mutable reference to running variance".into(),
            )
        })?;

        for c in 0..self.num_features {
            running_mean.data[c] =
                self.momentum * running_mean.data[c] + (1.0 - self.momentum) * mean[c];
            running_var.data[c] =
                self.momentum * running_var.data[c] + (1.0 - self.momentum) * var[c];
        }

        Ok(())
    }

    fn forward_impl(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        if input.shape.len() != 4 {
            return Err(BellandeError::InvalidShape(
                "Expected 4D tensor (batch_size, channels, height, width)".into(),
            ));
        }

        let (batch_size, channels, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

        if channels != self.num_features {
            return Err(BellandeError::InvalidOperation(format!(
                "Expected {} channels but got {}",
                self.num_features, channels
            )));
        }

        let mut output = input.data.clone();

        if self.training {
            let mut mean = vec![0.0; channels];
            let mut var = vec![0.0; channels];
            let size = batch_size * height * width;
            let n = size as f32;

            for c in 0..channels {
                let mut sum = 0.0;
                let mut sq_sum = 0.0;

                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            let val = input.data[idx];
                            sum += val;
                            sq_sum += val * val;
                        }
                    }
                }

                mean[c] = sum / n;
                var[c] = sq_sum / n - mean[c] * mean[c];
            }

            // Update running statistics
            self.update_running_stats(&mean, &var)?;

            // Normalize
            for c in 0..channels {
                let std = (var[c] + self.eps).sqrt();
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            output[idx] = (output[idx] - mean[c]) / std;

                            if let Some(ref weight) = self.weight {
                                output[idx] *= weight.data[c];
                            }
                            if let Some(ref bias) = self.bias {
                                output[idx] += bias.data[c];
                            }
                        }
                    }
                }
            }
        } else {
            // Use running statistics
            let running_mean = &self.running_mean;
            let running_var = &self.running_var;

            for c in 0..channels {
                let std = (running_var.data[c] + self.eps).sqrt();
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            output[idx] = (output[idx] - running_mean.data[c]) / std;

                            if let Some(ref weight) = self.weight {
                                output[idx] *= weight.data[c];
                            }
                            if let Some(ref bias) = self.bias {
                                output[idx] += bias.data[c];
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }
    fn backward_impl(&mut self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        let input = self.input.as_ref().ok_or(BellandeError::InvalidBackward(
            "Forward pass not called before backward".into(),
        ))?;

        let (batch_size, channels, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let spatial_size = height * width;
        let n = (batch_size * spatial_size) as f32;

        // Calculate mean and variance
        let mut mean = vec![0.0; channels];
        let mut var = vec![0.0; channels];

        for c in 0..channels {
            let mut sum = 0.0;
            let mut sq_sum = 0.0;
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        let val = input.data[idx];
                        sum += val;
                        sq_sum += val * val;
                    }
                }
            }
            mean[c] = sum / n;
            var[c] = sq_sum / n - mean[c] * mean[c];
        }

        // Initialize gradients
        let mut dx = vec![0.0; input.data.len()];
        let mut dweight = if self.weight.is_some() {
            vec![0.0; channels]
        } else {
            vec![]
        };
        let mut dbias = if self.bias.is_some() {
            vec![0.0; channels]
        } else {
            vec![]
        };

        // Compute gradients for each channel
        for c in 0..channels {
            let std = (var[c] + self.eps).sqrt();
            let inv_std = 1.0 / std;

            let mut sum_dxhat = 0.0;
            let mut sum_dxhat_x = 0.0;

            // First pass: compute sums for the channel
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        let xhat = (input.data[idx] - mean[c]) * inv_std;

                        let dxhat = grad_output.data[idx]
                            * if let Some(ref weight) = self.weight {
                                weight.data[c]
                            } else {
                                1.0
                            };

                        sum_dxhat += dxhat;
                        sum_dxhat_x += dxhat * xhat;
                    }
                }
            }

            // Second pass: compute dx for the channel
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let idx = ((b * channels + c) * height + h) * width + w;
                        let xhat = (input.data[idx] - mean[c]) * inv_std;

                        let dxhat = grad_output.data[idx]
                            * if let Some(ref weight) = self.weight {
                                weight.data[c]
                            } else {
                                1.0
                            };

                        dx[idx] = inv_std * (dxhat - sum_dxhat / n - xhat * sum_dxhat_x / n);
                    }
                }
            }

            // Compute dweight and dbias if they exist
            if let Some(_) = self.weight {
                dweight[c] = 0.0;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            let xhat = (input.data[idx] - mean[c]) * inv_std;
                            dweight[c] += grad_output.data[idx] * xhat;
                        }
                    }
                }
            }

            if let Some(_) = self.bias {
                dbias[c] = 0.0;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            dbias[c] += grad_output.data[idx];
                        }
                    }
                }
            }
        }

        // Update weight and bias gradients if they exist
        if let Some(ref mut weight) = self.weight {
            weight.grad = Some(dweight);
        }

        if let Some(ref mut bias) = self.bias {
            bias.grad = Some(dbias);
        }

        Ok(Tensor::new(
            dx,
            input.shape.clone(),
            true,
            input.device.clone(),
            input.dtype,
        ))
    }
}

// Implement NeuralLayer for BatchNorm1d
impl NeuralLayer for BatchNorm1d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let output = self.forward_impl(input)?;
        self.input = Some(input.clone());
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        self.backward_impl(grad_output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight.clone());
        }
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(("weight".to_string(), weight.clone()));
        }
        if let Some(ref bias) = self.bias {
            params.push(("bias".to_string(), bias.clone()));
        }
        params
    }

    fn set_parameter(&mut self, name: &str, value: Tensor) -> Result<(), BellandeError> {
        match name {
            "weight" => {
                if let Some(ref weight) = self.weight {
                    if value.shape == weight.shape {
                        self.weight = Some(value);
                        Ok(())
                    } else {
                        Err(BellandeError::ShapeMismatch("Weight shape mismatch".into()))
                    }
                } else {
                    Err(BellandeError::InvalidParameter(
                        "Layer does not use weights".into(),
                    ))
                }
            }
            "bias" => {
                if let Some(ref bias) = self.bias {
                    if value.shape == bias.shape {
                        self.bias = Some(value);
                        Ok(())
                    } else {
                        Err(BellandeError::ShapeMismatch("Bias shape mismatch".into()))
                    }
                } else {
                    Err(BellandeError::InvalidParameter(
                        "Layer does not use bias".into(),
                    ))
                }
            }
            _ => Err(BellandeError::InvalidParameter(format!(
                "Unknown parameter name: {}",
                name
            ))),
        }
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

// Implement NeuralLayer for BatchNorm2d
impl NeuralLayer for BatchNorm2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let output = self.forward_impl(input)?;
        self.input = Some(input.clone());
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        self.backward_impl(grad_output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight.clone());
        }
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(("weight".to_string(), weight.clone()));
        }
        if let Some(ref bias) = self.bias {
            params.push(("bias".to_string(), bias.clone()));
        }
        params
    }

    fn set_parameter(&mut self, name: &str, value: Tensor) -> Result<(), BellandeError> {
        match name {
            "weight" => {
                if let Some(ref weight) = self.weight {
                    if value.shape == weight.shape {
                        self.weight = Some(value);
                        Ok(())
                    } else {
                        Err(BellandeError::ShapeMismatch("Weight shape mismatch".into()))
                    }
                } else {
                    Err(BellandeError::InvalidParameter(
                        "Layer does not use weights".into(),
                    ))
                }
            }
            "bias" => {
                if let Some(ref bias) = self.bias {
                    if value.shape == bias.shape {
                        self.bias = Some(value);
                        Ok(())
                    } else {
                        Err(BellandeError::ShapeMismatch("Bias shape mismatch".into()))
                    }
                } else {
                    Err(BellandeError::InvalidParameter(
                        "Layer does not use bias".into(),
                    ))
                }
            }
            _ => Err(BellandeError::InvalidParameter(format!(
                "Unknown parameter name: {}",
                name
            ))),
        }
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for BatchNorm1d {}
unsafe impl Sync for BatchNorm1d {}
unsafe impl Send for BatchNorm2d {}
unsafe impl Sync for BatchNorm2d {}
