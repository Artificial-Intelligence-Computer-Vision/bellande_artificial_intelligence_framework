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

use crate::core::{device::Device, error::BellandeError};
use crate::data::dataloader::DataLoader;
use crate::models::models::Model;
use crate::training::{callbacks::Callback, history::TrainingHistory, validator::CallbackEvent};

use crate::loss::{
    bce::{BCELoss, Reduction},
    cross_entropy::CrossEntropyLoss,
    mse::MSELoss,
    Loss,
};

use crate::optim::{adam::Adam, rmsprop::RMSprop, scheduler::LRScheduler, sgd::SGD, Optimizer};
use std::collections::HashMap;

#[derive(Default)]
pub struct RunningMetrics {
    metrics: HashMap<String, (f32, usize)>,
}

impl RunningMetrics {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn update(&mut self, name: &str, value: f32) {
        let entry = self.metrics.entry(name.to_string()).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }

    pub fn get_average(&self) -> HashMap<String, f32> {
        self.metrics
            .iter()
            .map(|(k, (sum, count))| (k.clone(), sum / *count as f32))
            .collect()
    }

    pub fn get_current(&self) -> HashMap<String, f32> {
        self.get_average()
    }
}

pub struct Trainer {
    model: Box<dyn Model>,
    optimizer: Box<dyn Optimizer>,
    loss_fn: Box<dyn Loss>,
    device: Device,
    callbacks: Vec<Box<dyn Callback>>,
    history: TrainingHistory,
    scheduler: Option<Box<dyn LRScheduler>>,
}

impl Trainer {
    pub fn new(
        model: Box<dyn Model>,
        optimizer: Box<dyn Optimizer>,
        loss_fn: Box<dyn Loss>,
        device: Device,
    ) -> Self {
        Trainer {
            model,
            optimizer,
            loss_fn,
            device,
            callbacks: Vec::new(),
            history: TrainingHistory::new(),
            scheduler: None,
        }
    }

    pub fn new_with_adam(
        model: Box<dyn Model>,
        learning_rate: f32,
        device: Device,
    ) -> Result<Self, BellandeError> {
        let loss_fn = Box::new(MSELoss::new(Reduction::Mean));
        let optimizer = Box::new(Adam::new(
            model.parameters(),
            learning_rate,
            (0.9, 0.999),
            1e-8,
            0.0,
        ));

        Ok(Self::new(model, optimizer, loss_fn, device))
    }

    pub fn new_with_sgd(
        model: Box<dyn Model>,
        learning_rate: f32,
        momentum: f32,
        device: Device,
    ) -> Result<Self, BellandeError> {
        let loss_fn = Box::new(CrossEntropyLoss::new(Reduction::Mean, None, None));
        let optimizer = Box::new(SGD::new(
            model.parameters(),
            learning_rate,
            momentum,
            0.0,
            false,
        ));

        Ok(Self::new(model, optimizer, loss_fn, device))
    }

    pub fn new_with_rmsprop(
        model: Box<dyn Model>,
        learning_rate: f32,
        alpha: f32,
        device: Device,
    ) -> Result<Self, BellandeError> {
        let loss_fn = Box::new(BCELoss::new(Reduction::Mean, None));
        let optimizer = Box::new(RMSprop::new(
            model.parameters(),
            learning_rate,
            alpha,
            1e-8,
            0.0,
            0.0,
            false,
        ));

        Ok(Self::new(model, optimizer, loss_fn, device))
    }

    pub fn add_scheduler(&mut self, scheduler: Box<dyn LRScheduler>) {
        self.scheduler = Some(scheduler);
    }

    pub fn add_callback(&mut self, callback: Box<dyn Callback>) {
        self.callbacks.push(callback);
    }

    pub fn fit(
        &mut self,
        mut train_loader: DataLoader,
        mut val_loader: Option<DataLoader>,
        epochs: usize,
    ) -> Result<TrainingHistory, BellandeError> {
        let mut logs = HashMap::new();
        self.call_callbacks(CallbackEvent::TrainBegin, &logs)?;

        for epoch in 0..epochs {
            logs.clear();
            logs.insert("epoch".to_string(), epoch as f32);
            self.call_callbacks(CallbackEvent::EpochBegin, &logs)?;

            self.model.train();
            let train_metrics = self.train_epoch(&mut train_loader, epoch)?;
            logs.extend(train_metrics);

            if let Some(ref mut val_loader) = val_loader {
                self.model.eval();
                let val_metrics = self.validate(val_loader)?;
                logs.extend(
                    val_metrics
                        .into_iter()
                        .map(|(k, v)| (format!("val_{}", k), v)),
                );
            }

            if let Some(scheduler) = &mut self.scheduler {
                scheduler.step();
            }

            self.history.update(epoch, logs.clone());
            self.call_callbacks(CallbackEvent::EpochEnd, &logs)?;
        }

        self.call_callbacks(CallbackEvent::TrainEnd, &logs)?;
        Ok(self.history.clone())
    }

    fn train_epoch(
        &mut self,
        train_loader: &mut DataLoader,
        _epoch: usize,
    ) -> Result<HashMap<String, f32>, BellandeError> {
        let mut metrics = RunningMetrics::new();

        for batch in train_loader.iter() {
            let (data, target) = batch?;
            let batch_logs = HashMap::new();
            self.call_callbacks(CallbackEvent::BatchBegin, &batch_logs)?;

            let data = data.to_device(&self.device)?;
            let target = target.to_device(&self.device)?;

            // Forward pass
            let mut output = self.model.forward(&data)?;
            let loss = self.loss_fn.forward(&output, &target)?;

            // Backward pass
            self.optimizer.zero_grad();
            output.backward()?;
            self.optimizer.step()?;

            metrics.update("loss", loss.data[0]);

            let batch_logs = metrics.get_current();
            self.call_callbacks(CallbackEvent::BatchEnd, &batch_logs)?;
        }

        Ok(metrics.get_average())
    }

    fn validate(
        &mut self,
        val_loader: &mut DataLoader,
    ) -> Result<HashMap<String, f32>, BellandeError> {
        let mut metrics = RunningMetrics::new();

        for batch in val_loader.iter() {
            let (data, target) = batch?;
            let data = data.to_device(&self.device)?;
            let target = target.to_device(&self.device)?;
            let output = self.model.forward(&data)?;
            let loss = self.loss_fn.forward(&output, &target)?;
            metrics.update("loss", loss.data[0]);
        }

        Ok(metrics.get_average())
    }

    fn call_callbacks(
        &mut self,
        event: CallbackEvent,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        for callback in &mut self.callbacks {
            match event {
                CallbackEvent::TrainBegin => callback.on_train_begin(logs)?,
                CallbackEvent::TrainEnd => callback.on_train_end(logs)?,
                CallbackEvent::EpochBegin => {
                    callback.on_epoch_begin(logs.get("epoch").unwrap().clone() as usize, logs)?
                }
                CallbackEvent::EpochEnd => {
                    callback.on_epoch_end(logs.get("epoch").unwrap().clone() as usize, logs)?
                }
                CallbackEvent::BatchBegin => callback.on_batch_begin(0, logs)?,
                CallbackEvent::BatchEnd => callback.on_batch_end(0, logs)?,
            }
        }
        Ok(())
    }
}
