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
use crate::loss::bce::Reduction;
use crate::loss::Loss;

/// Cross Entropy Loss implementation with support for class weights and ignored indices
pub struct CrossEntropyLoss {
    reduction: Reduction,
    weight: Option<Tensor>,
    ignore_index: Option<i64>,
}

impl CrossEntropyLoss {
    pub fn new(reduction: Reduction, weight: Option<Tensor>, ignore_index: Option<i64>) -> Self {
        CrossEntropyLoss {
            reduction,
            weight,
            ignore_index,
        }
    }

    pub fn default() -> Self {
        CrossEntropyLoss {
            reduction: Reduction::Mean,
            weight: None,
            ignore_index: None,
        }
    }

    fn validate_input(&self, prediction: &Tensor, target: &Tensor) -> Result<(), BellandeError> {
        if prediction.shape.len() != 2 {
            return Err(BellandeError::InvalidInputs(
                "Prediction tensor must be 2-dimensional (batch_size, num_classes)".to_string(),
            ));
        }

        if target.shape.len() != 1 {
            return Err(BellandeError::InvalidInputs(
                "Target tensor must be 1-dimensional (batch_size)".to_string(),
            ));
        }

        if prediction.shape[0] != target.shape[0] {
            return Err(BellandeError::ShapeMismatch(
                "Batch sizes of prediction and target must match".to_string(),
            ));
        }

        Ok(())
    }

    fn compute_log_softmax(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let batch_size = input.shape[0];
        let num_classes = input.shape[1];

        // Find max values for numerical stability
        let mut max_vals = vec![f32::NEG_INFINITY; batch_size];
        for b in 0..batch_size {
            for c in 0..num_classes {
                let idx = b * num_classes + c;
                max_vals[b] = max_vals[b].max(input.data[idx]);
            }
        }

        // Compute exp(x - max) and sum
        let mut exp_sum = vec![0.0; batch_size];
        let mut shifted = vec![0.0; input.data.len()];

        for b in 0..batch_size {
            for c in 0..num_classes {
                let idx = b * num_classes + c;
                shifted[idx] = (input.data[idx] - max_vals[b]).exp();
                exp_sum[b] += shifted[idx];
            }
        }

        // Compute log_softmax
        let mut output = vec![0.0; input.data.len()];
        for b in 0..batch_size {
            let log_sum = exp_sum[b].ln();
            for c in 0..num_classes {
                let idx = b * num_classes + c;
                output[idx] = input.data[idx] - max_vals[b] - log_sum;
            }
        }

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            true,
            input.device.clone(),
            input.dtype,
        ))
    }

    fn compute_softmax(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let batch_size = input.shape[0];
        let num_classes = input.shape[1];

        // Find max values for numerical stability
        let mut max_vals = vec![f32::NEG_INFINITY; batch_size];
        for b in 0..batch_size {
            for c in 0..num_classes {
                let idx = b * num_classes + c;
                max_vals[b] = max_vals[b].max(input.data[idx]);
            }
        }

        // Compute exp(x - max) and sum
        let mut exp_sum = vec![0.0; batch_size];
        let mut output = vec![0.0; input.data.len()];

        for b in 0..batch_size {
            for c in 0..num_classes {
                let idx = b * num_classes + c;
                output[idx] = (input.data[idx] - max_vals[b]).exp();
                exp_sum[b] += output[idx];
            }
        }

        // Normalize
        for b in 0..batch_size {
            for c in 0..num_classes {
                let idx = b * num_classes + c;
                output[idx] /= exp_sum[b];
            }
        }

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            true,
            input.device.clone(),
            input.dtype,
        ))
    }

    fn convert_to_one_hot(
        &self,
        target: &Tensor,
        num_classes: usize,
    ) -> Result<Tensor, BellandeError> {
        let batch_size = target.shape[0];
        let mut one_hot = vec![0.0; batch_size * num_classes];

        for i in 0..batch_size {
            let target_idx = target.data[i] as usize;
            if target_idx >= num_classes {
                return Err(BellandeError::InvalidInputs(format!(
                    "Target class {} is out of range (0, {})",
                    target_idx,
                    num_classes - 1
                )));
            }
            one_hot[i * num_classes + target_idx] = 1.0;
        }

        Ok(Tensor::new(
            one_hot,
            vec![batch_size, num_classes],
            true,
            target.device.clone(),
            target.dtype,
        ))
    }

    fn element_wise_multiply(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BellandeError> {
        if a.shape != b.shape {
            return Err(BellandeError::ShapeMismatch(
                "Tensor shapes must match for multiplication".into(),
            ));
        }

        let output: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| x * y)
            .collect();

        Ok(Tensor::new(
            output,
            a.shape.clone(),
            true,
            a.device.clone(),
            a.dtype,
        ))
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        self.validate_input(prediction, target)?;

        let num_classes = prediction.shape[1];
        let log_softmax = self.compute_log_softmax(prediction)?;
        let target_one_hot = self.convert_to_one_hot(target, num_classes)?;

        // Compute negative log likelihood
        let mut loss = self.element_wise_multiply(&target_one_hot, &log_softmax)?;
        loss.data.iter_mut().for_each(|x| *x = -*x);

        // Apply class weights if provided
        if let Some(ref weight) = self.weight {
            loss = self.element_wise_multiply(&loss, weight)?;
        }

        // Apply ignore index masking if specified
        if let Some(ignore_idx) = self.ignore_index {
            for i in 0..target.shape[0] {
                if target.data[i] as i64 == ignore_idx {
                    for j in 0..loss.shape[1] {
                        loss.data[i * loss.shape[1] + j] = 0.0;
                    }
                }
            }
        }

        // Apply reduction
        match self.reduction {
            Reduction::Mean => {
                let sum: f32 = loss.data.iter().sum();
                let mean = sum / (loss.data.len() as f32);
                Ok(Tensor::new(
                    vec![mean],
                    vec![1],
                    true,
                    loss.device,
                    loss.dtype,
                ))
            }
            Reduction::Sum => {
                let sum: f32 = loss.data.iter().sum();
                Ok(Tensor::new(
                    vec![sum],
                    vec![1],
                    true,
                    loss.device,
                    loss.dtype,
                ))
            }
            Reduction::None => Ok(loss),
        }
    }

    fn backward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        let softmax = self.compute_softmax(prediction)?;
        let num_classes = prediction.shape[1];
        let target_one_hot = self.convert_to_one_hot(target, num_classes)?;

        // Compute gradients
        let mut grad_output = vec![0.0; softmax.data.len()];
        for i in 0..softmax.data.len() {
            grad_output[i] = softmax.data[i] - target_one_hot.data[i];
        }

        let mut grad = Tensor::new(
            grad_output,
            softmax.shape,
            true,
            softmax.device,
            softmax.dtype,
        );

        // Apply class weights if provided
        if let Some(ref weight) = self.weight {
            grad = self.element_wise_multiply(&grad, weight)?;
        }

        // Apply ignore index masking if specified
        if let Some(ignore_idx) = self.ignore_index {
            for i in 0..target.shape[0] {
                if target.data[i] as i64 == ignore_idx {
                    for j in 0..grad.shape[1] {
                        grad.data[i * grad.shape[1] + j] = 0.0;
                    }
                }
            }
        }

        // Apply reduction
        match self.reduction {
            Reduction::Mean => {
                let batch_size = prediction.shape[0] as f32;
                grad.data.iter_mut().for_each(|x| *x /= batch_size);
                Ok(grad)
            }
            _ => Ok(grad),
        }
    }
}

// Implement thread safety
unsafe impl Send for CrossEntropyLoss {}
unsafe impl Sync for CrossEntropyLoss {}
