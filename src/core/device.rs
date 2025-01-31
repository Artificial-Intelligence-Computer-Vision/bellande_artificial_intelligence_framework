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
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    CPU,
    CUDA(usize),
}

impl Device {
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::CUDA(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::CPU)
    }

    pub fn cuda_device_count() -> usize {
        #[cfg(feature = "cuda")]
        {
            match cuda_runtime::device_count() {
                Ok(count) => count,
                Err(_) => 0,
            }
        }
        #[cfg(not(feature = "cuda"))]
        0
    }

    pub fn default() -> Self {
        Device::CPU
    }

    pub fn from(device_str: &str) -> Result<Self, BellandeError> {
        Self::from_str(device_str)
    }

    #[cfg(feature = "cuda")]
    pub fn get_cuda_properties(
        device_id: usize,
    ) -> Result<cuda_runtime::DeviceProperties, BellandeError> {
        cuda_runtime::get_device_properties(device_id)
            .map_err(|_| BellandeError::DeviceNotAvailable)
    }

    #[cfg(feature = "cuda")]
    pub fn set_cuda_device(device_id: usize) -> Result<(), BellandeError> {
        cuda_runtime::set_device(device_id).map_err(|_| BellandeError::DeviceNotAvailable)
    }

    #[cfg(feature = "cuda")]
    pub fn get_current_cuda_device() -> Result<usize, BellandeError> {
        cuda_runtime::get_device().map_err(|_| BellandeError::DeviceNotAvailable)
    }

    #[cfg(feature = "cuda")]
    pub fn reset_cuda_device() -> Result<(), BellandeError> {
        cuda_runtime::device_reset().map_err(|_| BellandeError::DeviceNotAvailable)
    }

    pub fn get_device_name(&self) -> Result<String, BellandeError> {
        match self {
            Device::CPU => Ok("CPU".to_string()),
            Device::CUDA(device_id) => {
                #[cfg(feature = "cuda")]
                {
                    let props = Self::get_cuda_properties(*device_id)?;
                    Ok(props.name)
                }
                #[cfg(not(feature = "cuda"))]
                Err(BellandeError::NotImplemented(
                    "CUDA support not compiled".into(),
                ))
            }
        }
    }

    pub fn get_device_memory(&self) -> Result<usize, BellandeError> {
        match self {
            Device::CPU => {
                let sys_info = sys_info::mem_info().map_err(|_| {
                    BellandeError::SystemError("Failed to get system memory info".into())
                })?;
                Ok(sys_info.total as usize)
            }
            Device::CUDA(device_id) => {
                #[cfg(feature = "cuda")]
                {
                    let props = Self::get_cuda_properties(*device_id)?;
                    Ok(props.total_global_mem)
                }
                #[cfg(not(feature = "cuda"))]
                Err(BellandeError::NotImplemented(
                    "CUDA support not compiled".into(),
                ))
            }
        }
    }
}

impl FromStr for Device {
    type Err = BellandeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        if s == "cpu" {
            Ok(Device::CPU)
        } else if s.starts_with("cuda") {
            if s == "cuda" {
                Ok(Device::CUDA(0))
            } else {
                let parts: Vec<&str> = s.split(':').collect();
                if parts.len() != 2 {
                    return Err(BellandeError::InvalidDevice);
                }
                match parts[1].parse::<usize>() {
                    Ok(device_id) => {
                        if device_id < Self::cuda_device_count() {
                            Ok(Device::CUDA(device_id))
                        } else {
                            Err(BellandeError::DeviceNotAvailable)
                        }
                    }
                    Err(_) => Err(BellandeError::InvalidDevice),
                }
            }
        } else {
            Err(BellandeError::InvalidDevice)
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA(device_id) => write!(f, "cuda:{}", device_id),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::CPU
    }
}
