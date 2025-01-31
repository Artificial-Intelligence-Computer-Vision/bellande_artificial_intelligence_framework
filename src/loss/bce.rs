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
use crate::loss::Loss;
use std::f32;

#[derive(Debug, Clone, Copy)]
pub enum ReductionType {
    Sum,
    Mean,
    Max,
    Min,
    Product,
}

#[derive(Clone, Copy)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}

pub trait ReductionOps {
    fn reduce(&self, input: &Tensor) -> Result<Tensor, BellandeError>;
    fn reduce_backward(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError>;
}

#[derive(Debug)]
pub struct ReductionOperation {
    reduction_type: ReductionType,
    dim: Option<usize>,
    keepdim: bool,
    input_cache: Option<ReductionCache>,
}

pub struct BCELoss {
    reduction: Reduction,
    weight: Option<Tensor>,
    eps: f32,
}

#[derive(Debug)]
struct ReductionCache {
    input: Tensor,
    indices: Option<Vec<usize>>,
}

impl BCELoss {
    pub fn new(reduction: Reduction, weight: Option<Tensor>) -> Self {
        BCELoss {
            reduction,
            weight,
            eps: 1e-8,
        }
    }
}

// Implement Loss trait for BCELoss
impl Loss for BCELoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        if prediction.shape != target.shape {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut loss = Vec::with_capacity(prediction.data.len());
        for (pred, tgt) in prediction.data.iter().zip(target.data.iter()) {
            let p = pred.clamp(self.eps, 1.0 - self.eps);
            let l = -tgt * p.ln() - (1.0 - tgt) * (1.0 - p).ln();
            if let Some(ref weight) = self.weight {
                loss.push(l * weight.data[0]);
            } else {
                loss.push(l);
            }
        }

        match self.reduction {
            Reduction::None => Ok(Tensor::new(
                loss,
                prediction.shape.clone(),
                true,
                prediction.device.clone(),
                prediction.dtype,
            )),
            Reduction::Mean => Ok(Tensor::new(
                vec![loss.iter().sum::<f32>() / loss.len() as f32],
                vec![1],
                true,
                prediction.device.clone(),
                prediction.dtype,
            )),
            Reduction::Sum => Ok(Tensor::new(
                vec![loss.iter().sum()],
                vec![1],
                true,
                prediction.device.clone(),
                prediction.dtype,
            )),
        }
    }

    fn backward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        if prediction.shape != target.shape {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut grad = Vec::with_capacity(prediction.data.len());
        for (pred, tgt) in prediction.data.iter().zip(target.data.iter()) {
            let p = pred.clamp(self.eps, 1.0 - self.eps);
            let mut g = (p - tgt) / (p * (1.0 - p));

            if let Some(ref weight) = self.weight {
                g *= weight.data[0];
            }

            grad.push(g);
        }

        let grad = match self.reduction {
            Reduction::None => grad,
            Reduction::Mean => {
                let scale = 1.0 / prediction.data.len() as f32;
                grad.iter().map(|&g| g * scale).collect()
            }
            Reduction::Sum => grad,
        };

        Ok(Tensor::new(
            grad,
            prediction.shape.clone(),
            true,
            prediction.device.clone(),
            prediction.dtype,
        ))
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for BCELoss {}
unsafe impl Sync for BCELoss {}

impl ReductionOperation {
    pub fn new(reduction_type: ReductionType, dim: Option<usize>, keepdim: bool) -> Self {
        ReductionOperation {
            reduction_type,
            dim,
            keepdim,
            input_cache: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let (output_data, output_shape, indices) = match self.dim {
            Some(dim) => self.reduce_along_dim(input, dim)?,
            None => self.reduce_all(input)?,
        };

        self.input_cache = Some(ReductionCache {
            input: input.clone(),
            indices,
        });

        Ok(Tensor::new(
            output_data,
            output_shape,
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }

    pub fn backward(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        if let Some(ref cache) = self.input_cache {
            let input_shape = cache.input.shape.clone();
            let mut grad_input = vec![0.0; cache.input.data.len()];

            match self.reduction_type {
                ReductionType::Sum => {
                    self.backward_sum(&mut grad_input, grad_output, &input_shape)?;
                }
                ReductionType::Mean => {
                    self.backward_mean(&mut grad_input, grad_output, &input_shape)?;
                }
                ReductionType::Max | ReductionType::Min => {
                    self.backward_max_min(
                        &mut grad_input,
                        grad_output,
                        &cache.indices.clone().unwrap(),
                    )?;
                }
                ReductionType::Product => {
                    self.backward_product(&mut grad_input, grad_output, &cache.input)?;
                }
            }

            Ok(Tensor::new(
                grad_input,
                input_shape,
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

    fn reduce_along_dim(
        &self,
        input: &Tensor,
        dim: usize,
    ) -> Result<(Vec<f32>, Vec<usize>, Option<Vec<usize>>), BellandeError> {
        if dim >= input.shape.len() {
            return Err(BellandeError::RuntimeError(format!(
                "Dimension {} out of bounds for tensor of shape {:?}",
                dim, input.shape
            )));
        }

        let mut output_shape = input.shape.clone();
        if !self.keepdim {
            output_shape.remove(dim);
        } else {
            output_shape[dim] = 1;
        }

        let stride = input.shape[dim];
        let outer_size: usize = input.shape[..dim].iter().product();
        let inner_size: usize = input.shape[dim + 1..].iter().product();

        let mut output = Vec::new();
        let mut indices = if matches!(self.reduction_type, ReductionType::Max | ReductionType::Min)
        {
            Some(Vec::new())
        } else {
            None
        };

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut values = Vec::with_capacity(stride);
                for s in 0..stride {
                    let idx = (outer * stride + s) * inner_size + inner;
                    values.push(input.data[idx]);
                }

                let (result, index) = match self.reduction_type {
                    ReductionType::Sum => (values.iter().sum(), None),
                    ReductionType::Mean => (values.iter().sum::<f32>() / stride as f32, None),
                    ReductionType::Max => {
                        let (max_idx, &max_val) = values
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap();
                        (max_val, Some(max_idx))
                    }
                    ReductionType::Min => {
                        let (min_idx, &min_val) = values
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap();
                        (min_val, Some(min_idx))
                    }
                    ReductionType::Product => (values.iter().product(), None),
                };

                output.push(result);
                if let Some(ref mut indices_vec) = indices {
                    if let Some(idx) = index {
                        indices_vec.push(idx);
                    }
                }
            }
        }

        Ok((output, output_shape, indices))
    }

    fn reduce_all(
        &self,
        input: &Tensor,
    ) -> Result<(Vec<f32>, Vec<usize>, Option<Vec<usize>>), BellandeError> {
        let output_shape = if self.keepdim {
            vec![1; input.shape.len()]
        } else {
            vec![1]
        };

        let (result, indices) = match self.reduction_type {
            ReductionType::Sum => (vec![input.data.iter().sum()], None),
            ReductionType::Mean => (
                vec![input.data.iter().sum::<f32>() / input.data.len() as f32],
                None,
            ),
            ReductionType::Max => {
                let (max_idx, &max_val) = input
                    .data
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                (vec![max_val], Some(vec![max_idx]))
            }
            ReductionType::Min => {
                let (min_idx, &min_val) = input
                    .data
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                (vec![min_val], Some(vec![min_idx]))
            }
            ReductionType::Product => (vec![input.data.iter().product()], None),
        };

        Ok((result, output_shape, indices))
    }

    fn backward_sum(
        &self,
        grad_input: &mut [f32],
        grad_output: &Tensor,
        input_shape: &[usize],
    ) -> Result<(), BellandeError> {
        match self.dim {
            Some(dim) => {
                let stride = input_shape[dim];
                let outer_size: usize = input_shape[..dim].iter().product();
                let inner_size: usize = input_shape[dim + 1..].iter().product();

                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        let grad = grad_output.data[(outer * inner_size + inner)];
                        for s in 0..stride {
                            let idx = (outer * stride + s) * inner_size + inner;
                            grad_input[idx] = grad;
                        }
                    }
                }
            }
            None => {
                let grad = grad_output.data[0];
                grad_input.iter_mut().for_each(|x| *x = grad);
            }
        }
        Ok(())
    }

    fn backward_mean(
        &self,
        grad_input: &mut [f32],
        grad_output: &Tensor,
        input_shape: &[usize],
    ) -> Result<(), BellandeError> {
        match self.dim {
            Some(dim) => {
                let stride = input_shape[dim] as f32;
                let outer_size: usize = input_shape[..dim].iter().product();
                let inner_size: usize = input_shape[dim + 1..].iter().product();

                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        let grad = grad_output.data[(outer * inner_size + inner)] / stride;
                        for s in 0..input_shape[dim] {
                            let idx = (outer * input_shape[dim] + s) * inner_size + inner;
                            grad_input[idx] = grad;
                        }
                    }
                }
            }
            None => {
                let grad = grad_output.data[0] / grad_input.len() as f32;
                grad_input.iter_mut().for_each(|x| *x = grad);
            }
        }
        Ok(())
    }

    fn backward_max_min(
        &self,
        grad_input: &mut [f32],
        grad_output: &Tensor,
        indices: &[usize],
    ) -> Result<(), BellandeError> {
        for (idx, &grad) in indices.iter().zip(grad_output.data.iter()) {
            grad_input[*idx] = grad;
        }
        Ok(())
    }

    fn backward_product(
        &self,
        grad_input: &mut [f32],
        grad_output: &Tensor,
        input: &Tensor,
    ) -> Result<(), BellandeError> {
        match self.dim {
            Some(dim) => {
                let stride = input.shape[dim];
                let outer_size: usize = input.shape[..dim].iter().product();
                let inner_size: usize = input.shape[dim + 1..].iter().product();

                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        let mut product = 1.0;
                        for s in 0..stride {
                            let idx = (outer * stride + s) * inner_size + inner;
                            product *= input.data[idx];
                        }

                        let grad = grad_output.data[(outer * inner_size + inner)];
                        for s in 0..stride {
                            let idx = (outer * stride + s) * inner_size + inner;
                            grad_input[idx] = grad * product / input.data[idx];
                        }
                    }
                }
            }
            None => {
                let product: f32 = input.data.iter().product();
                let grad = grad_output.data[0];
                for (i, &val) in input.data.iter().enumerate() {
                    grad_input[i] = grad * product / val;
                }
            }
        }
        Ok(())
    }
}
