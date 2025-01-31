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

use crate::core::{
    autograd::AutogradFunction, device::Device, dtype::DataType, error::BellandeError,
};
use std::ops::{Add, Mul, Sub};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<f32>>,
    pub grad_fn: Option<Arc<dyn AutogradFunction>>,
    pub device: Device,
    pub dtype: DataType,
}

impl Tensor {
    pub fn new(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        device: Device,
        dtype: DataType,
    ) -> Self {
        let size = shape.iter().product();
        assert_eq!(data.len(), size, "Data size does not match shape");

        Tensor {
            data,
            shape,
            requires_grad,
            grad: if requires_grad {
                Some(vec![0.0; size])
            } else {
                None
            },
            grad_fn: None,
            device,
            dtype,
        }
    }

    // Data access methods
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    // Calculate stride for the current shape
    pub fn stride(&self) -> Vec<usize> {
        let mut stride = Vec::with_capacity(self.shape.len());
        let mut current_stride = 1;
        for &dim in self.shape.iter().rev() {
            stride.push(current_stride);
            current_stride *= dim;
        }
        stride.reverse();
        stride
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, BellandeError> {
        if self.shape != other.shape {
            return Err(BellandeError::ShapeMismatch(
                "Tensors must have the same shape for addition".into(),
            ));
        }

        let output: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(Tensor::new(
            output,
            self.shape.clone(),
            self.requires_grad || other.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Tensor, BellandeError> {
        if dims.len() != self.shape.len() {
            return Err(BellandeError::InvalidShape(format!(
                "Permutation dimensions must match tensor dimensions: expected {}, got {}",
                self.shape.len(),
                dims.len()
            )));
        }

        let mut new_shape = vec![0; self.shape.len()];
        for (i, &dim) in dims.iter().enumerate() {
            if dim >= self.shape.len() {
                return Err(BellandeError::InvalidShape(format!(
                    "Invalid permutation dimension: {}",
                    dim
                )));
            }
            new_shape[i] = self.shape[dim];
        }

        let mut new_data = vec![0.0; self.data.len()];
        let strides = self.compute_strides();
        let new_strides = compute_strides(&new_shape);

        for i in 0..self.data.len() {
            let old_indices = get_indices(i, &strides, &self.shape);
            let mut new_indices = vec![0; old_indices.len()];
            for (j, &dim) in dims.iter().enumerate() {
                new_indices[j] = old_indices[dim];
            }
            let new_idx = get_flat_index(&new_indices, &new_strides);
            new_data[new_idx] = self.data[i];
        }

        Ok(Tensor::new(
            new_data,
            new_shape,
            self.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }

    pub fn scale(&self, factor: f32) -> Result<Tensor, BellandeError> {
        let new_data = self.data.iter().map(|&x| x * factor).collect();

        Ok(Tensor::new(
            new_data,
            self.shape.clone(),
            self.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }

    fn compute_strides(&self) -> Vec<usize> {
        compute_strides(&self.shape)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor::new(
            vec![0.0; size],
            shape.to_vec(),
            false,
            Device::default(),
            DataType::default(),
        )
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor::new(
            vec![1.0; size],
            shape.to_vec(),
            false,
            Device::default(),
            DataType::default(),
        )
    }

    pub fn randn(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor::new(
            crate::core::random::normal(0.0, 1.0, size),
            shape.to_vec(),
            false,
            Device::default(),
            DataType::default(),
        )
    }

    pub fn stack(tensors: &[Tensor]) -> Result<Tensor, BellandeError> {
        if tensors.is_empty() {
            return Err(BellandeError::InvalidInputs(format!("Invalid Inputs")))?;
        }

        let base_shape = tensors[0].shape();

        // Verify all tensors have the same shape
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != base_shape {
                return Err(BellandeError::ShapeMismatch(format!(
                    "tensor 0 has shape {:?} but tensor {} has shape {:?}",
                    base_shape,
                    i,
                    tensor.shape()
                )));
            }
        }

        // Calculate new shape with batch dimension
        let mut new_shape = vec![tensors.len()];
        new_shape.extend(base_shape);

        // Calculate total size
        let total_size = new_shape.iter().product();
        let batch_size: usize = base_shape.iter().fold(1, |acc, &x| acc * x);
        let mut result_data = vec![0.0; total_size];

        // Copy data from each tensor
        for (i, tensor) in tensors.iter().enumerate() {
            let start = i * batch_size;
            let end = start + batch_size;
            result_data[start..end].copy_from_slice(&tensor.data);
        }

        Ok(Tensor::new(
            result_data,
            new_shape,
            tensors[0].requires_grad,
            tensors[0].device.clone(),
            tensors[0].dtype,
        ))
    }

    pub fn copy_slice(&mut self, batch_idx: usize, source: &Tensor) -> Result<(), BellandeError> {
        let strides = self.stride();
        if strides.is_empty() {
            return Err(BellandeError::InvalidShape("Empty tensor shape".into()));
        }

        let batch_stride = strides[0];
        let start_idx = batch_idx * batch_stride;
        let end_idx = start_idx + batch_stride;

        if end_idx > self.data.len() {
            return Err(BellandeError::IndexOutOfBounds);
        }

        // Check if source has correct size
        if source.data.len() != batch_stride {
            return Err(BellandeError::DimensionMismatch);
        }

        self.data[start_idx..end_idx].copy_from_slice(&source.data);
        Ok(())
    }

    pub fn backward(&mut self) -> Result<(), BellandeError> {
        if !self.requires_grad {
            return Err(BellandeError::NoGradients);
        }

        if self.grad.is_none() {
            self.grad = Some(vec![1.0; self.data.len()]);
        }

        if let Some(ref grad_fn) = self.grad_fn {
            if let Some(ref grad) = self.grad {
                grad_fn.backward(&Tensor::new(
                    grad.clone(),
                    self.shape.clone(),
                    false,
                    self.device.clone(),
                    self.dtype,
                ))?;
            }
        }

        Ok(())
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, BellandeError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(BellandeError::InvalidShape(
                "Tensors must be 2D for matmul".into(),
            ));
        }

        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k {
                    sum += self.data[i * k + k] * other.data[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Tensor::new(
            result,
            vec![m, n],
            self.requires_grad || other.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }

    pub fn to_device(&self, device: &Device) -> Result<Tensor, BellandeError> {
        Ok(Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            grad_fn: self.grad_fn.clone(),
            device: device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn t(&self) -> Result<Tensor, BellandeError> {
        if self.shape.len() != 2 {
            return Err(BellandeError::InvalidShape(
                "Transpose only works on 2D tensors".to_string(),
            ));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut transposed = vec![0.0; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Tensor {
            data: transposed,
            shape: vec![cols, rows],
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor, BellandeError> {
        if self.shape != mask.shape {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut new_data = self.data.clone();
        for (i, &mask_val) in mask.data.iter().enumerate() {
            if mask_val != 0.0 {
                new_data[i] = value;
            }
        }

        Ok(Tensor {
            data: new_data,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn softmax(&self, dim: i32) -> Result<Tensor, BellandeError> {
        let dim = if dim < 0 {
            (self.shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if dim >= self.shape.len() {
            return Err(BellandeError::RuntimeError(format!(
                "Dimension out of range (expected to be in range of [-{}, {}], but got {})",
                self.shape.len(),
                self.shape.len() - 1,
                dim
            )));
        }

        let mut result = vec![0.0; self.data.len()];
        let stride = self.get_stride(dim);
        let outer_size = self.shape[..dim].iter().product::<usize>();
        let inner_size = self.shape[dim + 1..].iter().product::<usize>();
        let dim_size = self.shape[dim];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for d in 0..dim_size {
                    let idx = outer * stride * dim_size + d * stride + inner;
                    max_val = max_val.max(self.data[idx]);
                }

                // Compute exponentials and sum
                let mut sum = 0.0;
                for d in 0..dim_size {
                    let idx = outer * stride * dim_size + d * stride + inner;
                    let exp_val = (self.data[idx] - max_val).exp();
                    result[idx] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for d in 0..dim_size {
                    let idx = outer * stride * dim_size + d * stride + inner;
                    result[idx] /= sum;
                }
            }
        }

        Ok(Tensor {
            data: result,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    // Helper method for softmax
    fn get_stride(&self, dim: usize) -> usize {
        let mut stride = 1;
        for d in dim + 1..self.shape.len() {
            stride *= self.shape[d];
        }
        stride
    }

    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor, BellandeError> {
        if dim >= self.shape.len() {
            return Err(BellandeError::InvalidShape(format!(
                "Dimension {} out of bounds",
                dim
            )));
        }

        let mut new_shape = self.shape.clone();
        if !keepdim {
            new_shape.remove(dim);
        } else {
            new_shape[dim] = 1;
        }

        let stride: usize = self.shape[dim..].iter().product();
        let outer_stride: usize = self.shape[..dim].iter().product();
        let inner_size: usize = stride / self.shape[dim];
        let mut result = vec![0.0; new_shape.iter().product()];

        for i in 0..outer_stride {
            for k in 0..inner_size {
                let mut sum = 0.0;
                for j in 0..self.shape[dim] {
                    let idx = i * stride + j * inner_size + k;
                    sum += self.data[idx];
                }
                result[i * inner_size + k] = sum;
            }
        }

        Ok(Tensor {
            data: result,
            shape: new_shape,
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn sum_all_dims(&self) -> Result<Tensor, BellandeError> {
        let sum = self.data.iter().sum();
        Ok(Tensor {
            data: vec![sum],
            shape: vec![1],
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor, BellandeError> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(BellandeError::InvalidShape(format!(
                "Cannot reshape tensor of size {} to shape {:?}",
                self.data.len(),
                new_shape
            )));
        }

        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            grad_fn: self.grad_fn.clone(),
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, BellandeError> {
        if self.shape != other.shape {
            return Err(BellandeError::ShapeMismatch(
                "Shapes must match for element-wise multiplication".into(),
            ));
        }

        let new_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Ok(Tensor::new(
            new_data,
            self.shape.clone(),
            self.requires_grad || other.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }

    pub fn transpose(&self) -> Result<Tensor, BellandeError> {
        if self.shape.len() != 2 {
            return Err(BellandeError::InvalidShape(
                "Transpose requires a 2D tensor".into(),
            ));
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut new_data = vec![0.0; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Tensor::new(
            new_data,
            vec![cols, rows],
            self.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }

    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor, BellandeError> {
        if dim >= self.shape.len() {
            return Err(BellandeError::InvalidShape(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.shape.len()
            )));
        }

        if start + length > self.shape[dim] {
            return Err(BellandeError::InvalidShape(
                "Narrow operation out of bounds".into(),
            ));
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = length;

        let mut new_data = Vec::new();
        let stride = self.get_stride(dim);

        // Collect the narrowed data
        for i in 0..self.data.len() {
            let dim_idx = (i / stride) % self.shape[dim];
            if dim_idx >= start && dim_idx < start + length {
                new_data.push(self.data[i]);
            }
        }

        Ok(Tensor::new(
            new_data,
            new_shape,
            self.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }

    // Hyperbolic tangent
    pub fn tanh(&self) -> Result<Tensor, BellandeError> {
        let new_data: Vec<f32> = self.data.iter().map(|&x| x.tanh()).collect();

        Ok(Tensor::new(
            new_data,
            self.shape.clone(),
            self.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }

    // Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Result<Tensor, BellandeError> {
        if self.shape != other.shape {
            return Err(BellandeError::ShapeMismatch(
                "Shapes must match for subtraction".into(),
            ));
        }

        let new_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(Tensor::new(
            new_data,
            self.shape.clone(),
            self.requires_grad || other.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn get_indices(flat_idx: usize, strides: &[usize], shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = flat_idx;
    for i in 0..shape.len() {
        indices[i] = remaining / strides[i];
        remaining %= strides[i];
    }
    indices
}

fn get_flat_index(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx * stride)
        .sum()
}

impl Add for &Tensor {
    type Output = Result<Tensor, BellandeError>;

    fn add(self, other: &Tensor) -> Self::Output {
        self.add(other)
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor, BellandeError>;

    fn mul(self, other: &Tensor) -> Self::Output {
        self.mul(other)
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor, BellandeError>;

    fn sub(self, other: &Tensor) -> Self::Output {
        self.sub(other)
    }
}
