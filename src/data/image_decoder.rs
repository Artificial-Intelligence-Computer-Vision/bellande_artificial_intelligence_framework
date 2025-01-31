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

use crate::core::{device::Device, dtype::DataType, error::BellandeError, tensor::Tensor};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Basic image format detector
#[derive(Debug, PartialEq)]
enum ImageFormat {
    JPEG,
    PNG,
    Unknown,
}

/// RGB pixel structure
#[derive(Clone, Copy, Debug)]
struct RGB {
    r: u8,
    g: u8,
    b: u8,
}

/// Image decoder implementation
pub struct ImageDecoder {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u8>,
}

impl ImageDecoder {
    /// Creates a new image decoder
    pub fn new(bytes: &[u8]) -> Result<Self, BellandeError> {
        let format = Self::detect_format(bytes)?;
        match format {
            ImageFormat::JPEG => Self::decode_jpeg(bytes),
            ImageFormat::PNG => Self::decode_png(bytes),
            ImageFormat::Unknown => Err(BellandeError::ImageError(
                "Unsupported image format".to_string(),
            )),
        }
    }

    /// Detects the image format from magic bytes
    fn detect_format(bytes: &[u8]) -> Result<ImageFormat, BellandeError> {
        if bytes.len() < 4 {
            return Err(BellandeError::ImageError("Invalid image data".to_string()));
        }

        match &bytes[0..4] {
            [0xFF, 0xD8, 0xFF, _] => Ok(ImageFormat::JPEG),
            [0x89, 0x50, 0x4E, 0x47] => Ok(ImageFormat::PNG),
            _ => Ok(ImageFormat::Unknown),
        }
    }

    /// Basic JPEG decoder implementation
    fn decode_jpeg(bytes: &[u8]) -> Result<Self, BellandeError> {
        // This is a basic implementation - you'll need to implement full JPEG decoding
        let mut reader = std::io::Cursor::new(bytes);
        let mut marker = [0u8; 2];

        // Find SOF0 marker (Start Of Frame)
        loop {
            reader.read_exact(&mut marker).map_err(|e| {
                BellandeError::ImageError(format!("Failed to read JPEG marker: {}", e))
            })?;

            if marker[0] != 0xFF {
                return Err(BellandeError::ImageError("Invalid JPEG marker".to_string()));
            }

            match marker[1] {
                0xC0 => break, // SOF0 marker
                0xD9 => return Err(BellandeError::ImageError("Reached end of JPEG".to_string())),
                _ => {
                    let mut length = [0u8; 2];
                    reader.read_exact(&mut length).map_err(|e| {
                        BellandeError::ImageError(format!("Failed to read length: {}", e))
                    })?;
                    let length = u16::from_be_bytes(length) as u64 - 2;
                    reader.set_position(reader.position() + length);
                }
            }
        }

        // Read image dimensions
        let mut header = [0u8; 5];
        reader
            .read_exact(&mut header)
            .map_err(|e| BellandeError::ImageError(format!("Failed to read SOF0 header: {}", e)))?;

        let height = u16::from_be_bytes([header[1], header[2]]) as usize;
        let width = u16::from_be_bytes([header[3], header[4]]) as usize;
        let channels = 3; // Assume RGB

        // Create placeholder data (you'll need to implement actual JPEG decoding)
        let data = vec![0u8; width * height * channels];

        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Basic PNG decoder implementation
    fn decode_png(bytes: &[u8]) -> Result<Self, BellandeError> {
        // This is a basic implementation - you'll need to implement full PNG decoding
        let mut reader = std::io::Cursor::new(bytes);
        let mut header = [0u8; 8];

        // Skip PNG signature
        reader
            .read_exact(&mut header)
            .map_err(|e| BellandeError::ImageError(format!("Failed to read PNG header: {}", e)))?;

        // Read IHDR chunk
        let mut length = [0u8; 4];
        reader.read_exact(&mut length).map_err(|e| {
            BellandeError::ImageError(format!("Failed to read chunk length: {}", e))
        })?;

        let mut ihdr = [0u8; 8];
        reader
            .read_exact(&mut ihdr)
            .map_err(|e| BellandeError::ImageError(format!("Failed to read IHDR: {}", e)))?;

        let width = u32::from_be_bytes([ihdr[0], ihdr[1], ihdr[2], ihdr[3]]) as usize;
        let height = u32::from_be_bytes([ihdr[4], ihdr[5], ihdr[6], ihdr[7]]) as usize;
        let channels = 3; // Assume RGB

        // Create placeholder data (you'll need to implement actual PNG decoding)
        let data = vec![0u8; width * height * channels];

        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Converts image data to tensor
    pub fn to_tensor(&self) -> Result<Tensor, BellandeError> {
        let mut tensor_data = Vec::with_capacity(self.width * self.height * self.channels);

        // Convert u8 to f32 and normalize to [0, 1]
        for &byte in &self.data {
            tensor_data.push(f32::from(byte) / 255.0);
        }

        Ok(Tensor::new(
            tensor_data,
            vec![1, self.channels, self.height, self.width],
            false,
            Device::CPU,
            DataType::Float32,
        ))
    }

    /// Resizes the image to specified dimensions
    pub fn resize(&mut self, new_width: usize, new_height: usize) -> Result<(), BellandeError> {
        if new_width == self.width && new_height == self.height {
            return Ok(());
        }

        let mut new_data = vec![0u8; new_width * new_height * self.channels];

        // Simple bilinear interpolation
        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 * self.width as f32 / new_width as f32).floor() as usize;
                let src_y = (y as f32 * self.height as f32 / new_height as f32).floor() as usize;

                for c in 0..self.channels {
                    let src_idx = (src_y * self.width + src_x) * self.channels + c;
                    let dst_idx = (y * new_width + x) * self.channels + c;
                    new_data[dst_idx] = self.data[src_idx];
                }
            }
        }

        self.width = new_width;
        self.height = new_height;
        self.data = new_data;

        Ok(())
    }
}

// Update ImageFolder implementation to use the decoder
pub struct ImageFolder {
    path: PathBuf,
    cache: HashMap<PathBuf, Tensor>,
    supported_extensions: Vec<String>,
}

impl ImageFolder {
    /// Creates a new ImageFolder instance
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, BellandeError> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(BellandeError::ImageError(format!(
                "Image folder does not exist: {}",
                path.display()
            )));
        }

        if !path.is_dir() {
            return Err(BellandeError::ImageError(format!(
                "Path is not a directory: {}",
                path.display()
            )));
        }

        Ok(Self {
            path,
            cache: HashMap::new(),
            supported_extensions: vec!["jpg".to_string(), "jpeg".to_string(), "png".to_string()],
        })
    }

    /// Decodes an image from bytes
    fn decode_image(bytes: &[u8]) -> Result<Tensor, BellandeError> {
        let mut decoder = ImageDecoder::new(bytes)?;

        // Resize to standard dimensions if needed
        if decoder.width != 224 || decoder.height != 224 {
            decoder.resize(224, 224)?;
        }

        decoder.to_tensor()
    }

    /// Loads an image from a file path
    pub fn load_image<P: AsRef<Path>>(&mut self, image_path: P) -> Result<Tensor, BellandeError> {
        let path = image_path.as_ref().to_path_buf();

        // Check cache first
        if let Some(tensor) = self.cache.get(&path) {
            return Ok(tensor.clone());
        }

        if !path.exists() {
            return Err(BellandeError::ImageError(format!(
                "Image file does not exist: {}",
                path.display()
            )));
        }

        // Verify file extension
        if let Some(ext) = path.extension() {
            if !self
                .supported_extensions
                .iter()
                .any(|e| e == &ext.to_string_lossy())
            {
                return Err(BellandeError::ImageError(format!(
                    "Unsupported image format: {}",
                    path.display()
                )));
            }
        }

        let bytes = std::fs::read(&path).map_err(|e| {
            BellandeError::ImageError(format!(
                "Failed to read image file {}: {}",
                path.display(),
                e
            ))
        })?;

        let tensor = Self::decode_image(&bytes)?;

        // Cache the result
        self.cache.insert(path, tensor.clone());

        Ok(tensor)
    }

    /// Lists all images in the folder
    pub fn list_images(&self) -> Result<Vec<PathBuf>, BellandeError> {
        let mut images = Vec::new();

        for entry in std::fs::read_dir(&self.path).map_err(|e| {
            BellandeError::ImageError(format!(
                "Failed to read directory {}: {}",
                self.path.display(),
                e
            ))
        })? {
            let entry = entry.map_err(|e| {
                BellandeError::ImageError(format!("Failed to read directory entry: {}", e))
            })?;

            let path = entry.path();

            if let Some(ext) = path.extension() {
                if self
                    .supported_extensions
                    .iter()
                    .any(|e| e == &ext.to_string_lossy())
                {
                    images.push(path);
                }
            }
        }

        Ok(images)
    }

    /// Clears the image cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Gets the base path of the image folder
    pub fn path(&self) -> &Path {
        &self.path
    }
}
