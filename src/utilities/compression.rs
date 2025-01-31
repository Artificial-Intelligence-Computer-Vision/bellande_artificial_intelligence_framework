// Copyright (C) 2025 Bellande Artificial Intelligence Computer Vision Research Innovation Center, Ronaldson Bellande

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

use std::cmp::min;
use std::io::{self, Read};

const WINDOW_SIZE: usize = 32768;
const WINDOW_MASK: usize = WINDOW_SIZE - 1;
const MAX_BITS: usize = 15;
const END_BLOCK: u16 = 256;

// Huffman code lengths for fixed literal/length tree
const FIXED_LITERAL_LENGTHS: &[u8] = &[
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 0-15
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 16-31
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 32-47
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 48-63
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 64-79
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 80-95
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 96-111
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 112-127
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 128-143
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 144-159
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 160-175
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 176-191
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 192-207
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 208-223
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 224-239
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 240-255
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // 256-271
    7, 7, 7, 7, 7, 7, 7, 7, // 272-279
    8, 8, 8, 8, 8, 8, 8, 8, // 280-287
];

// Length and distance base values and extra bits
const LENGTH_CODES: &[(u16, u8)] = &[
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 1),
    (13, 1),
    (15, 1),
    (17, 1),
    (19, 2),
    (23, 2),
    (27, 2),
    (31, 2),
    (35, 3),
    (43, 3),
    (51, 3),
    (59, 3),
    (67, 4),
    (83, 4),
    (99, 4),
    (115, 4),
    (131, 5),
    (163, 5),
    (195, 5),
    (227, 5),
    (258, 0),
];

const DISTANCE_CODES: &[(u16, u8)] = &[
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 1),
    (7, 1),
    (9, 2),
    (13, 2),
    (17, 3),
    (25, 3),
    (33, 4),
    (49, 4),
    (65, 5),
    (97, 5),
    (129, 6),
    (193, 6),
    (257, 7),
    (385, 7),
    (513, 8),
    (769, 8),
    (1025, 9),
    (1537, 9),
    (2049, 10),
    (3073, 10),
    (4097, 11),
    (6145, 11),
    (8193, 12),
    (12289, 12),
    (16385, 13),
    (24577, 13),
];

#[derive(Clone)]
struct HuffmanTree {
    counts: Vec<u16>,
    symbols: Vec<u16>,
    min_code: Vec<u16>,
    max_code: Vec<u16>,
}

impl HuffmanTree {
    fn new() -> Self {
        HuffmanTree {
            counts: vec![0; MAX_BITS + 1],
            symbols: Vec::new(),
            min_code: vec![0; MAX_BITS + 1],
            max_code: vec![0; MAX_BITS + 1],
        }
    }

    fn build_from_lengths(&mut self, lengths: &[u8], max_symbol: usize) -> io::Result<()> {
        // Count the number of codes for each code length
        self.counts.fill(0);
        for &len in lengths.iter().take(max_symbol) {
            if len as usize > MAX_BITS {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid code length",
                ));
            }
            self.counts[len as usize] += 1;
        }

        // Compute first code value for each code length
        let mut code = 0;
        self.min_code[0] = 0;
        self.max_code[0] = 0;
        for bits in 1..=MAX_BITS {
            code = (code + self.counts[bits - 1]) << 1;
            self.min_code[bits] = code;
            self.max_code[bits] = code + self.counts[bits] - 1;
        }

        // Assign symbols to codes
        self.symbols = vec![0; max_symbol];
        let mut symbol_index = 0;
        for bits in 1..=MAX_BITS {
            for symbol in 0..max_symbol {
                if lengths[symbol] as usize == bits {
                    self.symbols[symbol_index] = symbol as u16;
                    symbol_index += 1;
                }
            }
        }

        Ok(())
    }

    fn decode_symbol<R: Read>(
        &self,
        reader: &mut R,
        bit_reader: &mut BitReader,
    ) -> io::Result<u16> {
        let mut len = 1;
        let mut code = 0;

        while len <= MAX_BITS {
            code = (code << 1) | if bit_reader.read_bit(reader)? { 1 } else { 0 };

            if code <= self.max_code[len] {
                let index = (code - self.min_code[len]) as usize;
                if index < self.symbols.len() {
                    return Ok(self.symbols[index]);
                }
            }
            len += 1;
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Huffman code",
        ))
    }
}

struct BitReader {
    bit_buffer: u32,
    bits_in_buffer: u8,
}

impl BitReader {
    fn new() -> Self {
        BitReader {
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    fn read_bit<R: Read>(&mut self, reader: &mut R) -> io::Result<bool> {
        if self.bits_in_buffer == 0 {
            let mut byte = [0u8; 1];
            reader.read_exact(&mut byte)?;
            self.bit_buffer = byte[0] as u32;
            self.bits_in_buffer = 8;
        }
        let bit = self.bit_buffer & 1 == 1;
        self.bit_buffer >>= 1;
        self.bits_in_buffer -= 1;
        Ok(bit)
    }

    fn read_bits<R: Read>(&mut self, reader: &mut R, mut count: u8) -> io::Result<u32> {
        let mut result = 0;
        let mut bits_read = 0;

        while bits_read < count {
            if self.bits_in_buffer == 0 {
                let mut byte = [0u8; 1];
                reader.read_exact(&mut byte)?;
                self.bit_buffer = byte[0] as u32;
                self.bits_in_buffer = 8;
            }

            let bits_to_take = min(count - bits_read, self.bits_in_buffer);
            let mask = (1 << bits_to_take) - 1;
            result |= ((self.bit_buffer & mask) << bits_read) as u32;

            self.bit_buffer >>= bits_to_take;
            self.bits_in_buffer -= bits_to_take;
            bits_read += bits_to_take;
        }

        Ok(result)
    }
}

pub struct Decoder<R> {
    inner: R,
    window: Vec<u8>,
    window_pos: usize,
    output_buffer: Vec<u8>,
    output_pos: usize,
    literal_tree: HuffmanTree,
    distance_tree: HuffmanTree,
    bit_reader: BitReader,
}

impl<R: Read> Decoder<R> {
    pub fn new(inner: R) -> Self {
        Decoder {
            inner,
            window: vec![0; WINDOW_SIZE],
            window_pos: 0,
            output_buffer: Vec::new(),
            output_pos: 0,
            literal_tree: HuffmanTree::new(),
            distance_tree: HuffmanTree::new(),
            bit_reader: BitReader::new(),
        }
    }

    fn read_header(&mut self) -> io::Result<()> {
        let mut header = [0u8; 2];
        self.inner.read_exact(&mut header)?;

        let cmf = header[0];
        let flg = header[1];

        if (cmf & 0x0F) != 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid compression method",
            ));
        }

        if (((cmf as u16) << 8) | flg as u16) % 31 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid header checksum",
            ));
        }

        Ok(())
    }

    fn decode_literal_symbol(&mut self) -> io::Result<u16> {
        self.literal_tree
            .decode_symbol(&mut self.inner, &mut self.bit_reader)
    }

    fn decode_distance_symbol(&mut self) -> io::Result<u16> {
        self.distance_tree
            .decode_symbol(&mut self.inner, &mut self.bit_reader)
    }

    fn process_block(&mut self) -> io::Result<bool> {
        let is_final = self.bit_reader.read_bit(&mut self.inner)?;
        let block_type = self.bit_reader.read_bits(&mut self.inner, 2)? as u8;

        match block_type {
            0 => self.decode_uncompressed_block()?,
            1 => {
                self.literal_tree
                    .build_from_lengths(FIXED_LITERAL_LENGTHS, 288)?;
                let distance_lengths = vec![5u8; 32];
                self.distance_tree
                    .build_from_lengths(&distance_lengths, 32)?;
                self.process_huffman_block()?;
            }
            2 => {
                self.decode_dynamic_huffman_block()?;
                self.process_huffman_block()?;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid block type",
                ))
            }
        }

        Ok(is_final)
    }

    fn decode_dynamic_huffman_block(&mut self) -> io::Result<()> {
        let hlit = self.bit_reader.read_bits(&mut self.inner, 5)? as usize + 257;
        let hdist = self.bit_reader.read_bits(&mut self.inner, 5)? as usize + 1;
        let hclen = self.bit_reader.read_bits(&mut self.inner, 4)? as usize + 4;

        let cl_index = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        let mut cl_lengths = vec![0u8; 19];
        for i in 0..hclen {
            cl_lengths[cl_index[i]] = self.bit_reader.read_bits(&mut self.inner, 3)? as u8;
        }

        let mut code_length_tree = HuffmanTree::new();
        code_length_tree.build_from_lengths(&cl_lengths, 19)?;

        let mut lengths = Vec::with_capacity(hlit + hdist);
        while lengths.len() < hlit + hdist {
            let symbol = code_length_tree.decode_symbol(&mut self.inner, &mut self.bit_reader)?;
            match symbol {
                0..=15 => lengths.push(symbol as u8),
                16 => {
                    if lengths.is_empty() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Invalid code lengths",
                        ));
                    }
                    let repeat = self.bit_reader.read_bits(&mut self.inner, 2)? as usize + 3;
                    let value = *lengths.last().unwrap();
                    lengths.extend(std::iter::repeat(value).take(repeat));
                }
                17 => {
                    let repeat = self.bit_reader.read_bits(&mut self.inner, 3)? as usize + 3;
                    lengths.extend(std::iter::repeat(0u8).take(repeat));
                }
                18 => {
                    let repeat = self.bit_reader.read_bits(&mut self.inner, 7)? as usize + 11;
                    lengths.extend(std::iter::repeat(0u8).take(repeat));
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid code length code",
                    ))
                }
            }
        }

        let (literal_lengths, distance_lengths) = lengths.split_at(hlit);
        self.literal_tree
            .build_from_lengths(literal_lengths, hlit)?;
        self.distance_tree
            .build_from_lengths(distance_lengths, hdist)?;

        Ok(())
    }

    fn process_huffman_block(&mut self) -> io::Result<()> {
        loop {
            let symbol = self
                .literal_tree
                .decode_symbol(&mut self.inner, &mut self.bit_reader)?;

            if symbol == END_BLOCK {
                break;
            }

            if symbol < 256 {
                // Literal byte
                self.window[self.window_pos] = symbol as u8;
                self.window_pos = (self.window_pos + 1) & WINDOW_MASK;
                self.output_buffer.push(symbol as u8);
            } else {
                // Length/distance pair
                let length = self.decode_length(symbol as usize - 257)?;
                let distance_code = self.decode_distance_symbol()?;
                let distance = self.decode_distance(distance_code as usize)?;

                if distance > self.window_pos {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }

                let start_pos = (self.window_pos - distance) & WINDOW_MASK;
                for i in 0..length {
                    let byte = self.window[(start_pos + i) & WINDOW_MASK];
                    self.window[self.window_pos] = byte;
                    self.window_pos = (self.window_pos + 1) & WINDOW_MASK;
                    self.output_buffer.push(byte);
                }
            }
        }

        Ok(())
    }

    fn decode_length(&mut self, code: usize) -> io::Result<usize> {
        if code >= LENGTH_CODES.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid length code",
            ));
        }

        let (base, extra) = LENGTH_CODES[code];
        let extra_bits = if extra > 0 {
            self.bit_reader.read_bits(&mut self.inner, extra)? as usize
        } else {
            0
        };

        Ok(base as usize + extra_bits)
    }

    fn decode_distance(&mut self, code: usize) -> io::Result<usize> {
        if code >= DISTANCE_CODES.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance code",
            ));
        }

        let (base, extra) = DISTANCE_CODES[code];
        let extra_bits = if extra > 0 {
            self.bit_reader.read_bits(&mut self.inner, extra)? as usize
        } else {
            0
        };

        Ok(base as usize + extra_bits)
    }

    fn decode_uncompressed_block(&mut self) -> io::Result<()> {
        // Reset bit buffer since we'll be reading byte-aligned data
        self.bit_reader.bits_in_buffer = 0;

        let mut header = [0u8; 4];
        self.inner.read_exact(&mut header)?;

        let len = u16::from_le_bytes([header[0], header[1]]);
        let nlen = u16::from_le_bytes([header[2], header[3]]);

        if len != !nlen {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid block length",
            ));
        }

        let mut buffer = vec![0; len as usize];
        self.inner.read_exact(&mut buffer)?;

        for &byte in &buffer {
            self.window[self.window_pos] = byte;
            self.window_pos = (self.window_pos + 1) & WINDOW_MASK;
            self.output_buffer.push(byte);
        }

        Ok(())
    }
}

impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.window_pos == 0 {
            self.read_header()?;
        }

        if self.output_pos < self.output_buffer.len() {
            let remaining = self.output_buffer.len() - self.output_pos;
            let to_copy = min(remaining, buf.len());
            buf[..to_copy]
                .copy_from_slice(&self.output_buffer[self.output_pos..self.output_pos + to_copy]);
            self.output_pos += to_copy;
            return Ok(to_copy);
        }

        self.output_pos = 0;
        self.output_buffer.clear();

        let is_final = self.process_block()?;

        if self.output_pos < self.output_buffer.len() {
            self.read(buf)
        } else if is_final {
            Ok(0)
        } else {
            self.read(buf)
        }
    }
}
