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
use crate::models::models::Model;
use crate::optim::Optimizer;
use std::sync::{Arc, Mutex};

pub struct DistributedTrainer {
    world_size: usize,
    rank: usize,
    model: Arc<Mutex<Box<dyn Model>>>,
    optimizer: Arc<Mutex<Box<dyn Optimizer>>>,
    loss_fn: Box<dyn Loss>,
}

impl DistributedTrainer {
    pub fn new(
        model: Box<dyn Model>,
        optimizer: Box<dyn Optimizer>,
        loss_fn: Box<dyn Loss>,
        world_size: usize,
        rank: usize,
    ) -> Self {
        DistributedTrainer {
            world_size,
            rank,
            model: Arc::new(Mutex::new(model)),
            optimizer: Arc::new(Mutex::new(optimizer)),
            loss_fn,
        }
    }

    pub async fn average_gradients(&self) {
        let model = self.model.lock().expect("Failed to lock model mutex");
        let mut optimizer = self
            .optimizer
            .lock()
            .expect("Failed to lock optimizer mutex");

        for param in optimizer.parameters_mut() {
            if let Some(grad) = &param.grad {
                let averaged_grad: Vec<f32> =
                    grad.iter().map(|&g| g / self.world_size as f32).collect();
                param.grad = Some(averaged_grad);
            }
        }
    }

    pub async fn train_step(&self, batch: (Tensor, Tensor)) -> Result<f32, BellandeError> {
        let (batch_x, batch_y) = batch;
        let loss_tensor;
        {
            let mut model = self.model.lock().expect("Failed to lock model mutex");
            let mut optimizer = self
                .optimizer
                .lock()
                .expect("Failed to lock optimizer mutex");
            optimizer.zero_grad();
            let prediction = model.forward(&batch_x)?;
            loss_tensor = self.loss_fn.forward(&prediction, &batch_y)?;
            let grad = self.loss_fn.backward(&prediction, &batch_y)?;
            model.backward(&grad)?;
        }

        self.average_gradients().await;
        let mut optimizer = self
            .optimizer
            .lock()
            .expect("Failed to lock optimizer mutex");
        optimizer.step();

        let loss_value = loss_tensor.data()[0];
        Ok(loss_value)
    }
}
