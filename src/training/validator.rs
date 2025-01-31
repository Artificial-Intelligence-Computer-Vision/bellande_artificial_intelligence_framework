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
use crate::metrics::metrics::Metric;
use crate::models::models::Model;
use std::collections::HashMap;

pub struct Validator {
    model: Box<dyn Model>,
    metrics: Vec<Box<dyn Metric>>,
    device: Device,
}

impl Validator {
    pub fn new(model: Box<dyn Model>, metrics: Vec<Box<dyn Metric>>, device: Device) -> Self {
        Validator {
            model,
            metrics,
            device,
        }
    }

    pub fn validate(
        &mut self,
        val_loader: &mut DataLoader,
    ) -> Result<HashMap<String, f32>, BellandeError> {
        self.model.eval();

        // Reset all metrics at the start of validation
        for metric in &mut self.metrics {
            metric.reset();
        }

        for batch in val_loader.iter() {
            let (data, target) = batch?;

            // Move data to device
            let data = data.to_device(&self.device)?;
            let target = target.to_device(&self.device)?;

            let output = self.model.forward(&data)?;

            // Update each metric with the current batch
            for metric in &mut self.metrics {
                metric.update(&output, &target);
            }
        }

        // Compute final metrics
        let mut results = HashMap::new();
        for metric in &self.metrics {
            results.insert(metric.name().to_string(), metric.compute());
        }

        Ok(results)
    }
}

pub enum CallbackEvent {
    TrainBegin,
    TrainEnd,
    EpochBegin,
    EpochEnd,
    BatchBegin,
    BatchEnd,
}
