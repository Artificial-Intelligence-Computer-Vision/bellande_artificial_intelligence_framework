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

use crate::core::error::BellandeError;
use crate::core::tensor::Tensor;
use crate::data::{dataset::Dataset, sampler::Sampler};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub struct DataLoader {
    dataset: Arc<Box<dyn Dataset>>,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
    sampler: Option<Arc<Mutex<Box<dyn Sampler>>>>,
    drop_last: bool,
}

impl DataLoader {
    pub fn new(
        dataset: Box<dyn Dataset>,
        batch_size: usize,
        shuffle: bool,
        num_workers: usize,
        sampler: Option<Box<dyn Sampler>>,
        drop_last: bool,
    ) -> Self {
        DataLoader {
            dataset: Arc::new(dataset),
            batch_size,
            shuffle,
            num_workers,
            sampler: sampler.map(|s| Arc::new(Mutex::new(s))),
            drop_last,
        }
    }

    pub fn iter(&self) -> DataLoaderIterator {
        DataLoaderIterator {
            dataloader: self,
            index: 0,
        }
    }
}

pub struct DataLoaderIterator<'a> {
    dataloader: &'a DataLoader,
    index: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = Result<(Tensor, Tensor), BellandeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataloader.dataset.len() {
            return None;
        }

        let batch_indices: Vec<usize> = if let Some(sampler) = &self.dataloader.sampler {
            match sampler.lock() {
                Ok(mut sampler) => sampler.sample(self.dataloader.batch_size),
                Err(_) => return Some(Err(BellandeError::LockError)),
            }
        } else if self.dataloader.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..self.dataloader.dataset.len()).collect();
            indices.shuffle(&mut rng);
            indices[..self.dataloader.batch_size.min(indices.len())].to_vec()
        } else {
            let end = (self.index + self.dataloader.batch_size).min(self.dataloader.dataset.len());
            (self.index..end).collect()
        };

        if batch_indices.is_empty()
            || (self.dataloader.drop_last && batch_indices.len() < self.dataloader.batch_size)
        {
            return None;
        }

        let batch: Vec<(Tensor, Tensor)> = if self.dataloader.num_workers > 1 {
            batch_indices
                .par_iter()
                .map(|&idx| self.dataloader.dataset.get(idx))
                .collect()
        } else {
            batch_indices
                .iter()
                .map(|&idx| self.dataloader.dataset.get(idx))
                .collect()
        };

        self.index += self.dataloader.batch_size;

        if batch.is_empty() {
            None
        } else {
            Some(collate_batch(batch))
        }
    }
}

fn get_batch_shape(tensors: &[Tensor]) -> Result<Vec<usize>, BellandeError> {
    if tensors.is_empty() {
        return Err(BellandeError::InvalidInputs(
            "Empty tensor batch".to_string(),
        ));
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

    // Create the batch shape: [batch_size, ...base_shape]
    let mut batch_shape = vec![tensors.len()];
    batch_shape.extend(base_shape);
    Ok(batch_shape)
}

fn collate_batch(batch: Vec<(Tensor, Tensor)>) -> Result<(Tensor, Tensor), BellandeError> {
    if batch.is_empty() {
        return Err(BellandeError::InvalidInputs(
            "Empty batch provided".to_string(),
        ));
    }

    // Split the batch into data and labels
    let (data_tensors, label_tensors): (Vec<Tensor>, Vec<Tensor>) = batch.into_iter().unzip();

    // Get shapes for data and labels
    let data_shape = get_batch_shape(&data_tensors)?;
    let label_shape = get_batch_shape(&label_tensors)?;

    // Create storage for batched data
    let mut batched_data = Tensor::zeros(&data_shape);
    let mut batched_labels = Tensor::zeros(&label_shape);

    // Copy data into the batched tensor
    for (i, data) in data_tensors.iter().enumerate() {
        copy_tensor_slice(&mut batched_data, i, data)?;
    }

    // Copy labels into the batched tensor
    for (i, label) in label_tensors.iter().enumerate() {
        copy_tensor_slice(&mut batched_labels, i, label)?;
    }

    Ok((batched_data, batched_labels))
}

fn copy_tensor_slice(
    dest: &mut Tensor,
    batch_idx: usize,
    source: &Tensor,
) -> Result<(), BellandeError> {
    let batch_stride = dest.stride()[0];
    let start_idx = batch_idx * batch_stride;
    let end_idx = start_idx + batch_stride;

    if end_idx > dest.data().len() {
        return Err(BellandeError::IndexOutOfBounds);
    }

    if source.data().len() != batch_stride {
        return Err(BellandeError::DimensionMismatch);
    }

    let dest_slice = &mut dest.data_mut()[start_idx..end_idx];
    dest_slice.copy_from_slice(source.data());
    Ok(())
}
