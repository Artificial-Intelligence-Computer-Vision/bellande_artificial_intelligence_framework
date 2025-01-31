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
use crate::data::augmentation::Transform;
use crate::utilities::byte::{BigEndian, ReadBytes};
use crate::utilities::compression::Decoder;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;

/// Implementation of From trait for error conversion
impl From<std::io::Error> for BellandeError {
    fn from(error: std::io::Error) -> Self {
        BellandeError::IOError(error.to_string())
    }
}

/// A reader that allows reading individual bits from a byte stream
pub struct BitReader<R: Read> {
    reader: R,
    buffer: u8,
    bits_remaining: u8,
}

/// Image format enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
enum ImageFormat {
    JPEG,
    PNG,
    Unknown,
}

#[derive(Debug, Clone, Copy)]
struct RGBPixel {
    r: u8,
    g: u8,
    b: u8,
}

impl RGBPixel {
    fn new(r: u8, g: u8, b: u8) -> Self {
        RGBPixel { r, g, b }
    }
}

/// Trait defining the interface for datasets
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Result<(Tensor, Tensor), BellandeError>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn num_classes(&self) -> usize;
}

/// Structure for managing image datasets organized in folders
pub struct ImageFolder {
    root: PathBuf,
    samples: Vec<(PathBuf, usize)>,
    transform: Option<Box<dyn Transform>>,
    target_transform: Option<Box<dyn Transform>>,
    class_to_idx: HashMap<String, usize>,
    cache: Option<RwLock<HashMap<PathBuf, Arc<Tensor>>>>,
    cache_size: usize,
}

impl<R: Read> BitReader<R> {
    /// Creates a new BitReader from a byte stream
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: 0,
            bits_remaining: 0,
        }
    }

    /// Reads a single bit from the stream
    pub fn read_bit(&mut self) -> io::Result<bool> {
        if self.bits_remaining == 0 {
            let mut byte = [0u8; 1];
            self.reader.read_exact(&mut byte)?;
            self.buffer = byte[0];
            self.bits_remaining = 8;
        }

        self.bits_remaining -= 1;
        Ok(((self.buffer >> self.bits_remaining) & 1) == 1)
    }

    /// Reads multiple bits and returns them as a u32
    pub fn read_bits(&mut self, mut count: u8) -> io::Result<u32> {
        let mut result = 0u32;

        while count > 0 {
            result = (result << 1) | (if self.read_bit()? { 1 } else { 0 });
            count -= 1;
        }

        Ok(result)
    }
}

impl ImageFolder {
    const JPEG_NATURAL_ORDER: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    /// Creates a new ImageFolder dataset
    pub fn new(
        root: PathBuf,
        transform: Option<Box<dyn Transform>>,
        target_transform: Option<Box<dyn Transform>>,
    ) -> Result<Self, BellandeError> {
        let mut samples = Vec::new();
        let mut class_to_idx = HashMap::new();

        Self::validate_root_directory(&root)?;
        Self::scan_directory(&root, &mut samples, &mut class_to_idx)?;

        if samples.is_empty() {
            return Err(BellandeError::IOError("No valid images found".to_string()));
        }

        Ok(ImageFolder {
            root,
            samples,
            transform,
            target_transform,
            class_to_idx,
            cache: Some(RwLock::new(HashMap::new())),
            cache_size: 1000,
        })
    }

    /// Creates a new ImageFolder with specified cache size
    pub fn with_cache_size(
        root: PathBuf,
        transform: Option<Box<dyn Transform>>,
        target_transform: Option<Box<dyn Transform>>,
        cache_size: usize,
    ) -> Result<Self, BellandeError> {
        let mut folder = Self::new(root, transform, target_transform)?;
        folder.cache_size = cache_size;
        Ok(folder)
    }

    /// Validates the root directory exists and is a directory
    fn validate_root_directory(root: &PathBuf) -> Result<(), BellandeError> {
        if !root.exists() || !root.is_dir() {
            return Err(BellandeError::IOError("Invalid root directory".to_string()));
        }
        Ok(())
    }

    /// Scans the directory structure and builds the dataset
    fn scan_directory(
        root: &PathBuf,
        samples: &mut Vec<(PathBuf, usize)>,
        class_to_idx: &mut HashMap<String, usize>,
    ) -> Result<(), BellandeError> {
        for (idx, entry) in fs::read_dir(root)?.enumerate() {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let class_name = path
                    .file_name()
                    .ok_or_else(|| {
                        BellandeError::IOError("Invalid class directory name".to_string())
                    })?
                    .to_string_lossy()
                    .into_owned();

                class_to_idx.insert(class_name, idx);
                Self::scan_images(&path, idx, samples)?;
            }
        }
        Ok(())
    }

    /// Scans for images in a directory
    fn scan_images(
        path: &PathBuf,
        class_idx: usize,
        samples: &mut Vec<(PathBuf, usize)>,
    ) -> Result<(), BellandeError> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && Self::is_valid_image(&path) {
                samples.push((path, class_idx));
            } else if path.is_dir() {
                Self::scan_images(&path, class_idx, samples)?;
            }
        }
        Ok(())
    }

    /// Checks if a file is a valid image based on its extension and header
    fn is_valid_image(path: &PathBuf) -> bool {
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
                if let Ok(bytes) = Self::read_image_file(path) {
                    return Self::detect_image_format(&bytes) != ImageFormat::Unknown;
                }
            }
        }
        false
    }

    /// Reads an image file to bytes
    fn read_image_file(path: &PathBuf) -> Result<Vec<u8>, BellandeError> {
        let mut file = File::open(path)
            .map_err(|e| BellandeError::IOError(format!("Failed to open image file: {}", e)))?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| BellandeError::IOError(format!("Failed to read image file: {}", e)))?;

        Ok(bytes)
    }

    /// Detects image format from bytes
    fn detect_image_format(bytes: &[u8]) -> ImageFormat {
        if bytes.len() < 4 {
            return ImageFormat::Unknown;
        }

        match &bytes[0..4] {
            [0xFF, 0xD8, 0xFF, _] => ImageFormat::JPEG,
            [0x89, 0x50, 0x4E, 0x47] => ImageFormat::PNG,
            _ => ImageFormat::Unknown,
        }
    }

    /// Decodes image bytes to RGB pixels
    fn decode_image_to_rgb(bytes: &[u8]) -> Result<(Vec<RGBPixel>, usize, usize), BellandeError> {
        match Self::detect_image_format(bytes) {
            ImageFormat::JPEG => Self::decode_jpeg(bytes),
            ImageFormat::PNG => Self::decode_png(bytes),
            ImageFormat::Unknown => Err(BellandeError::ImageError(
                "Unknown image format".to_string(),
            )),
        }
    }

    fn decode_jpeg(bytes: &[u8]) -> Result<(Vec<RGBPixel>, usize, usize), BellandeError> {
        let mut cursor = Cursor::new(bytes);
        let mut marker = [0u8; 2];

        // Verify JPEG signature (0xFFD8)
        cursor
            .read_exact(&mut marker)
            .map_err(|e| BellandeError::ImageError(format!("Invalid JPEG header: {}", e)))?;

        if marker != [0xFF, 0xD8] {
            return Err(BellandeError::ImageError(
                "Not a valid JPEG file".to_string(),
            ));
        }

        let mut width = 0;
        let mut height = 0;
        let mut components = 0;
        let mut quantization_tables: HashMap<u8, Vec<u8>> = HashMap::new();
        let mut huffman_tables: HashMap<(u8, u8), Vec<u8>> = HashMap::new();

        loop {
            cursor.read_exact(&mut marker)?;

            if marker[0] != 0xFF {
                return Err(BellandeError::ImageError("Invalid marker".to_string()));
            }

            match marker[1] {
                0xC0 => {
                    // Start of Frame
                    let mut segment = [0u8; 8];
                    cursor.read_exact(&mut segment)?;

                    let precision = segment[0];
                    height = u16::from_be_bytes([segment[1], segment[2]]) as usize;
                    width = u16::from_be_bytes([segment[3], segment[4]]) as usize;
                    components = segment[5] as usize;

                    if precision != 8 {
                        return Err(BellandeError::ImageError(
                            "Only 8-bit precision supported".to_string(),
                        ));
                    }

                    let mut comp_info = vec![0u8; components * 3];
                    cursor.read_exact(&mut comp_info)?;
                }

                0xDB => {
                    // Define Quantization Table
                    let mut length_bytes = [0u8; 2];
                    cursor.read_exact(&mut length_bytes)?;
                    let length = u16::from_be_bytes(length_bytes) as usize - 2;

                    let mut table_data = vec![0u8; length];
                    cursor.read_exact(&mut table_data)?;

                    let precision = (table_data[0] >> 4) & 0x0F;
                    let table_id = table_data[0] & 0x0F;
                    let table_size = if precision == 0 { 64 } else { 128 };

                    quantization_tables.insert(table_id, table_data[1..=table_size].to_vec());
                }

                0xC4 => {
                    // Define Huffman Table
                    let mut length_bytes = [0u8; 2];
                    cursor.read_exact(&mut length_bytes)?;
                    let length = u16::from_be_bytes(length_bytes) as usize - 2;

                    let mut table_data = vec![0u8; length];
                    cursor.read_exact(&mut table_data)?;

                    let table_class = (table_data[0] >> 4) & 0x0F;
                    let table_id = table_data[0] & 0x0F;

                    let mut codes = Vec::new();
                    let mut offset = 17;
                    for &length in &table_data[1..17] {
                        for _ in 0..length {
                            codes.push(table_data[offset]);
                            offset += 1;
                        }
                    }

                    huffman_tables.insert((table_class, table_id), codes);
                }

                0xDA => {
                    // Start of Scan
                    let mut length_bytes = [0u8; 2];
                    cursor.read_exact(&mut length_bytes)?;
                    let length = u16::from_be_bytes(length_bytes) as usize - 2;

                    let mut scan_data = vec![0u8; length];
                    cursor.read_exact(&mut scan_data)?;

                    // Process compressed data
                    let mut pixels = vec![RGBPixel::new(0, 0, 0); width * height];
                    let mut bit_reader = BitReader::new(&mut cursor);

                    // Process MCUs (Minimum Coded Units)
                    let mcu_width = ((width + 7) / 8) * 8;
                    let mcu_height = ((height + 7) / 8) * 8;

                    for y in (0..mcu_height).step_by(8) {
                        for x in (0..mcu_width).step_by(8) {
                            for component in 0..components {
                                let component_u8 = component as u8;
                                let qtable = &quantization_tables[&component_u8];
                                let (dc_table, ac_table) = (
                                    &huffman_tables[&(0u8, component_u8)],
                                    &huffman_tables[&(1u8, component_u8)],
                                );

                                let block = Self::decode_block(
                                    &mut bit_reader,
                                    dc_table,
                                    ac_table,
                                    qtable,
                                )?;

                                if component == 0 {
                                    for by in 0..8 {
                                        for bx in 0..8 {
                                            let px = x + bx;
                                            let py = y + by;
                                            if px < width && py < height {
                                                let idx = py * width + px;
                                                pixels[idx].r = block[by * 8 + bx] as u8;
                                                pixels[idx].g = block[by * 8 + bx] as u8;
                                                pixels[idx].b = block[by * 8 + bx] as u8;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    return Ok((pixels, width, height));
                }

                0xD9 => break, // End of Image

                _ => {
                    // Skip other markers
                    let mut length_bytes = [0u8; 2];
                    cursor.read_exact(&mut length_bytes)?;
                    let length = u16::from_be_bytes(length_bytes) as usize - 2;
                    cursor.seek(SeekFrom::Current(length as i64))?;
                }
            }
        }

        Err(BellandeError::ImageError(
            "Failed to decode JPEG".to_string(),
        ))
    }

    fn decode_block(
        bit_reader: &mut BitReader<impl Read>,
        dc_table: &[u8],
        ac_table: &[u8],
        qtable: &[u8],
    ) -> Result<Vec<u8>, BellandeError> {
        const BLOCK_SIZE: usize = 64;
        let mut block = vec![0u8; BLOCK_SIZE];
        let mut zz = [0i32; BLOCK_SIZE];

        // Decode DC coefficient
        let dc_value = Self::decode_huffman_value(bit_reader, dc_table)?;
        if dc_value > 0 {
            let bits = Self::receive_and_extend(bit_reader, dc_value as u8)?;
            zz[0] = bits;
        }

        // Decode AC coefficients
        let mut k = 1;
        while k < BLOCK_SIZE {
            let rs = Self::decode_huffman_value(bit_reader, ac_table)?;
            let s = rs & 0x0F;
            let r = rs >> 4;

            if s == 0 {
                if r == 15 {
                    k += 16; // Skip 16 zeros
                    continue;
                }
                break; // End of block
            }

            k += r as usize; // Skip zeros
            if k >= BLOCK_SIZE {
                return Err(BellandeError::ImageError(
                    "Invalid AC coefficient index".to_string(),
                ));
            }

            // Read additional bits
            let value = Self::receive_and_extend(bit_reader, s as u8)?;
            zz[Self::JPEG_NATURAL_ORDER[k]] = value;
            k += 1;
        }

        // Dequantize
        for i in 0..BLOCK_SIZE {
            zz[i] *= qtable[i] as i32;
        }

        // Inverse DCT
        Self::inverse_dct(&mut zz);

        // Level shift and clamp values
        for i in 0..BLOCK_SIZE {
            let val = ((zz[i] + 128) >> 8).clamp(0, 255);
            block[i] = val as u8;
        }

        Ok(block)
    }

    fn decode_huffman_value(
        bit_reader: &mut BitReader<impl Read>,
        table: &[u8],
    ) -> Result<u8, BellandeError> {
        let mut code = 0;
        let mut code_len = 0;
        let mut index = 0;

        loop {
            code = (code << 1)
                | if bit_reader
                    .read_bit()
                    .map_err(|e| BellandeError::ImageError(e.to_string()))?
                {
                    1
                } else {
                    0
                };
            code_len += 1;

            while index < table.len() && table[index] as u8 == code_len {
                if code as u8 == table[index + 1] {
                    return Ok(table[index + 2]);
                }
                index += 3;
            }

            if code_len >= 16 {
                return Err(BellandeError::ImageError(
                    "Invalid Huffman code".to_string(),
                ));
            }
        }
    }

    fn receive_and_extend(
        bit_reader: &mut BitReader<impl Read>,
        nbits: u8,
    ) -> Result<i32, BellandeError> {
        if nbits == 0 {
            return Ok(0);
        }

        let value = bit_reader
            .read_bits(nbits)
            .map_err(|e| BellandeError::ImageError(e.to_string()))? as i32;

        let vt = 1 << (nbits - 1);
        Ok(if value < vt {
            value + (-1 << nbits) + 1
        } else {
            value
        })
    }

    fn inverse_dct(block: &mut [i32; 64]) {
        // Constants for IDCT
        const W1: i32 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
        const W2: i32 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
        const W3: i32 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
        const W5: i32 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
        const W6: i32 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
        const W7: i32 = 565; // 2048*sqrt(2)*cos(7*pi/16)

        let mut tmp = [0i32; 64];

        // Row IDCT
        for i in 0..8 {
            let row_offset = i * 8;
            let x0 = block[row_offset];
            let x1 = block[row_offset + 4];
            let x2 = block[row_offset + 2];
            let x3 = block[row_offset + 6];
            let x4 = block[row_offset + 1];
            let x5 = block[row_offset + 5];
            let x6 = block[row_offset + 3];
            let x7 = block[row_offset + 7];

            // Stage 1
            let x8 = W7 * (x4 + x5);
            let x4 = x8 + (W1 - W7) * x4;
            let x5 = x8 - (W1 + W7) * x5;
            let x8 = W3 * (x6 + x7);
            let x6 = x8 - (W3 - W5) * x6;
            let x7 = x8 - (W3 + W5) * x7;

            // Stage 2
            let x8 = x0 + x1;
            let x0 = x0 - x1;
            let x1 = W6 * (x2 + x3);
            let x2 = x1 - (W2 + W6) * x3;
            let x3 = x1 + (W2 - W6) * x2;

            // Stage 3
            let x1 = x4 + x6;
            let x4 = x4 - x6;
            let x6 = x5 + x7;
            let x5 = x5 - x7;

            // Stage 4
            let x7 = x8 + x3;
            let x8_final = x8 - x3; // Renamed to avoid shadowing
            let x3 = x0 + x2;
            let x0 = x0 - x2;
            let x2 = (181 * (x4 + x5) + 128) >> 8;
            let x4 = (181 * (x4 - x5) + 128) >> 8;

            // Output
            tmp[row_offset] = (x7 + x1) >> 3;
            tmp[row_offset + 1] = (x3 + x2) >> 3;
            tmp[row_offset + 2] = (x0 + x4) >> 3;
            tmp[row_offset + 3] = (x8_final + x6) >> 3;
            tmp[row_offset + 4] = (x8_final - x6) >> 3;
            tmp[row_offset + 5] = (x0 - x4) >> 3;
            tmp[row_offset + 6] = (x3 - x2) >> 3;
            tmp[row_offset + 7] = (x7 - x1) >> 3;
        }

        // Column IDCT
        for i in 0..8 {
            let x0 = tmp[i];
            let x1 = tmp[i + 32];
            let x2 = tmp[i + 16];
            let x3 = tmp[i + 48];
            let x4 = tmp[i + 8];
            let x5 = tmp[i + 40];
            let x6 = tmp[i + 24];
            let x7 = tmp[i + 56];

            // Stage 1
            let x8 = W7 * (x4 + x5);
            let x4 = x8 + (W1 - W7) * x4;
            let x5 = x8 - (W1 + W7) * x5;
            let x8 = W3 * (x6 + x7);
            let x6 = x8 - (W3 - W5) * x6;
            let x7 = x8 - (W3 + W5) * x7;

            // Stage 2
            let x8 = x0 + x1;
            let x0 = x0 - x1;
            let x1 = W6 * (x2 + x3);
            let x2 = x1 - (W2 + W6) * x3;
            let x3 = x1 + (W2 - W6) * x2;

            // Stage 3
            let x1 = x4 + x6;
            let x4 = x4 - x6;
            let x6 = x5 + x7;
            let x5 = x5 - x7;

            // Stage 4
            let x7 = x8 + x3;
            let x8_final = x8 - x3;
            let x3 = x0 + x2;
            let x0 = x0 - x2;
            let x2 = (181 * (x4 + x5) + 128) >> 8;
            let x4 = (181 * (x4 - x5) + 128) >> 8;

            // Final output with proper scaling
            block[i] = (x7 + x1) >> 14;
            block[i + 8] = (x3 + x2) >> 14;
            block[i + 16] = (x0 + x4) >> 14;
            block[i + 24] = (x8_final + x6) >> 14;
            block[i + 32] = (x8_final - x6) >> 14;
            block[i + 40] = (x0 - x4) >> 14;
            block[i + 48] = (x3 - x2) >> 14;
            block[i + 56] = (x7 - x1) >> 14;
        }
    }

    /// Decodes PNG image bytes
    fn decode_png(bytes: &[u8]) -> Result<(Vec<RGBPixel>, usize, usize), BellandeError> {
        let mut cursor = Cursor::new(bytes);

        // Verify PNG signature
        let mut signature = [0u8; 8];
        cursor.read_exact(&mut signature).map_err(|e| {
            BellandeError::ImageError(format!("Failed to read PNG signature: {}", e))
        })?;

        if signature != [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
            return Err(BellandeError::ImageError(
                "Invalid PNG signature".to_string(),
            ));
        }

        let mut width = 0;
        let mut height = 0;
        let mut image_data = Vec::new();
        let mut palette = Vec::new();
        let mut bit_depth = 0;
        let mut color_type = 0;

        loop {
            let length = cursor.read_u32::<BigEndian>().map_err(|e| {
                BellandeError::ImageError(format!("Failed to read chunk length: {}", e))
            })? as usize;

            let mut chunk_type = [0u8; 4];
            cursor.read_exact(&mut chunk_type).map_err(|e| {
                BellandeError::ImageError(format!("Failed to read chunk type: {}", e))
            })?;

            match &chunk_type {
                b"IHDR" => {
                    width = cursor.read_u32::<BigEndian>().map_err(|e| {
                        BellandeError::ImageError(format!("Failed to read width: {}", e))
                    })? as usize;
                    height = cursor.read_u32::<BigEndian>().map_err(|e| {
                        BellandeError::ImageError(format!("Failed to read height: {}", e))
                    })? as usize;

                    let mut ihdr_data = [0u8; 5];
                    cursor.read_exact(&mut ihdr_data).map_err(|e| {
                        BellandeError::ImageError(format!("Failed to read IHDR data: {}", e))
                    })?;

                    bit_depth = ihdr_data[0];
                    color_type = ihdr_data[1];

                    cursor.seek(SeekFrom::Current(4))?; // Skip CRC
                }

                b"PLTE" => {
                    palette = vec![0u8; length];
                    cursor.read_exact(&mut palette).map_err(|e| {
                        BellandeError::ImageError(format!("Failed to read palette: {}", e))
                    })?;
                    cursor.seek(SeekFrom::Current(4))?; // Skip CRC
                }

                b"IDAT" => {
                    let mut chunk_data = vec![0u8; length];
                    cursor.read_exact(&mut chunk_data).map_err(|e| {
                        BellandeError::ImageError(format!("Failed to read IDAT chunk: {}", e))
                    })?;
                    image_data.extend(chunk_data);
                    cursor.seek(SeekFrom::Current(4))?; // Skip CRC
                }

                b"IEND" => break,

                _ => {
                    cursor
                        .seek(SeekFrom::Current((length + 4) as i64))
                        .map_err(|e| {
                            BellandeError::ImageError(format!("Failed to skip chunk: {}", e))
                        })?;
                }
            }
        }

        // Process image data based on color type
        let mut decoder = Decoder::new(&image_data[..]);
        let mut decoded_data = Vec::new();
        decoder.read_to_end(&mut decoded_data)?;

        let pixels = match color_type {
            2 => {
                // RGB
                let bpp = 3;
                let stride = width * bpp + 1;
                let mut pixels = Vec::with_capacity(width * height);

                for y in 0..height {
                    let row_start = y * stride + 1; // Skip filter byte
                    for x in 0..width {
                        let i = row_start + x * bpp;
                        pixels.push(RGBPixel::new(
                            decoded_data[i],
                            decoded_data[i + 1],
                            decoded_data[i + 2],
                        ));
                    }
                }
                pixels
            }

            3 => {
                // Palette
                if palette.is_empty() {
                    return Err(BellandeError::ImageError(
                        "Missing palette data".to_string(),
                    ));
                }

                let stride = width + 1;
                let mut pixels = Vec::with_capacity(width * height);

                for y in 0..height {
                    let row_start = y * stride + 1; // Skip filter byte
                    for x in 0..width {
                        let index = (decoded_data[row_start + x] as usize) * 3;
                        pixels.push(RGBPixel::new(
                            palette[index],
                            palette[index + 1],
                            palette[index + 2],
                        ));
                    }
                }
                pixels
            }

            6 => {
                // RGBA
                let bpp = 4;
                let stride = width * bpp + 1;
                let mut pixels = Vec::with_capacity(width * height);

                for y in 0..height {
                    let row_start = y * stride + 1; // Skip filter byte
                    for x in 0..width {
                        let i = row_start + x * bpp;
                        pixels.push(RGBPixel::new(
                            decoded_data[i],
                            decoded_data[i + 1],
                            decoded_data[i + 2],
                        ));
                    }
                }
                pixels
            }

            _ => {
                return Err(BellandeError::ImageError(format!(
                    "Unsupported color type: {}",
                    color_type
                )))
            }
        };

        Ok((pixels, width, height))
    }

    /// Converts RGB pixels to tensor
    fn rgb_to_tensor(
        pixels: &[RGBPixel],
        width: usize,
        height: usize,
    ) -> Result<Tensor, BellandeError> {
        if pixels.len() != width * height {
            return Err(BellandeError::ImageError(format!(
                "Invalid pixel buffer size: expected {}, got {}",
                width * height,
                pixels.len()
            )));
        }

        let mut data = Vec::with_capacity(3 * width * height);

        // Convert to CHW format and normalize to [0, 1]
        for channel in 0..3 {
            data.extend(pixels.iter().map(|pixel| {
                let value = match channel {
                    0 => pixel.r,
                    1 => pixel.g,
                    2 => pixel.b,
                    _ => unreachable!(),
                };
                f32::from(value) / 255.0
            }));
        }

        Ok(Tensor::new(
            data,
            vec![1, 3, height, width],
            false,
            Device::CPU,
            DataType::Float32,
        ))
    }

    /// Gets a cached tensor or loads it from disk
    fn get_cached_tensor(&self, path: &PathBuf) -> Result<Arc<Tensor>, BellandeError> {
        if let Some(cache_lock) = &self.cache {
            // Try to read from cache first
            if let Ok(cache) = cache_lock.read() {
                if let Some(tensor) = cache.get(path) {
                    return Ok(Arc::clone(tensor));
                }
            }

            // Not in cache, load it
            let bytes = Self::read_image_file(path)?;
            let (pixels, width, height) = Self::decode_image_to_rgb(&bytes)?;
            let tensor = Arc::new(Self::rgb_to_tensor(&pixels, width, height)?);

            // Update cache
            if let Ok(mut cache) = cache_lock.write() {
                // Manage cache size
                if cache.len() >= self.cache_size {
                    if let Some(key) = cache.keys().next().cloned() {
                        cache.remove(&key);
                    }
                }
                cache.insert(path.clone(), Arc::clone(&tensor));
            }

            Ok(tensor)
        } else {
            // Cache disabled, just load and return
            let bytes = Self::read_image_file(path)?;
            let (pixels, width, height) = Self::decode_image_to_rgb(&bytes)?;
            Ok(Arc::new(Self::rgb_to_tensor(&pixels, width, height)?))
        }
    }

    pub fn num_classes(&self) -> usize {
        self.class_to_idx.len()
    }

    pub fn get_class_to_idx(&self) -> &HashMap<String, usize> {
        &self.class_to_idx
    }

    pub fn get_sample_path(&self, index: usize) -> Option<&PathBuf> {
        self.samples.get(index).map(|(path, _)| path)
    }

    pub fn set_caching(&mut self, enabled: bool) {
        self.cache = if enabled {
            Some(RwLock::new(HashMap::new()))
        } else {
            None
        };
    }

    pub fn clear_cache(&self) {
        if let Some(cache_lock) = &self.cache {
            if let Ok(mut cache) = cache_lock.write() {
                cache.clear();
            }
        }
    }
}

impl Dataset for ImageFolder {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn num_classes(&self) -> usize {
        self.num_classes()
    }

    fn get(&self, index: usize) -> Result<(Tensor, Tensor), BellandeError> {
        let (path, class_idx) = &self.samples[index];
        let input = self.get_cached_tensor(path)?;

        let target = Tensor::new(
            vec![*class_idx as f32],
            vec![1],
            false,
            input.get_device().clone(),
            input.get_dtype().clone(),
        );

        let mut final_input = (*input).clone();
        if let Some(transform) = &self.transform {
            final_input = transform.apply(&final_input)?;
        }

        let mut final_target = target;
        if let Some(target_transform) = &self.target_transform {
            final_target = target_transform.apply(&final_target)?;
        }

        Ok((final_input, final_target))
    }
}
