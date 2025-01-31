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
use std::error::Error;
use std::path::Path;

pub mod core;
pub mod data;
pub mod layer;
pub mod loss;
pub mod metrics;
pub mod models;
pub mod optim;
pub mod training;
pub mod utilities;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const FRAMEWORK_NAME: &str = "Bellande AI Framework";

pub struct Framework {
    config: utilities::config::Configuration,
    device: Device,
    initialized: bool,
}

impl Framework {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let default_config = utilities::config::Configuration::default();
        Ok(Framework {
            config: default_config,
            device: Device::CPU,
            initialized: false,
        })
    }

    pub fn with_config<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn Error>> {
        let config = utilities::config::Configuration::from_file(config_path)?;
        let device = Device::from(&config.system.device)?;

        Ok(Framework {
            config,
            device,
            initialized: false,
        })
    }

    pub fn initialize(&mut self) -> Result<(), Box<dyn Error>> {
        if self.initialized {
            return Ok(());
        }

        // Set random seed if specified
        if let Some(seed) = self.config.system.seed {
            core::random::set_seed(seed);
        }

        // Initialize CUDA if available and requested
        if self.device.is_cuda() {
            #[cfg(feature = "cuda")]
            {
                if Device::cuda_device_count() == 0 {
                    return Err(Box::new(BellandeError::DeviceNotAvailable));
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Box::new(BellandeError::NotImplemented(
                    "CUDA support not compiled".into(),
                )));
            }
        }

        self.initialized = true;
        Ok(())
    }

    pub fn get_version() -> &'static str {
        VERSION
    }

    pub fn get_name() -> &'static str {
        FRAMEWORK_NAME
    }

    pub fn system_info() -> String {
        format!(
            "{} v{}\n\
            CPU Threads: {}\n\
            CUDA Available: {}\n\
            CUDA Devices: {}\n\
            Default Device: {}",
            FRAMEWORK_NAME,
            VERSION,
            num_cpus::get(),
            cfg!(feature = "cuda"),
            Device::cuda_device_count(),
            Device::default(),
        )
    }
}
