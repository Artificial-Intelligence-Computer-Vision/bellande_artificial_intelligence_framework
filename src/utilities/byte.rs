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

use std::io::{self, Read};

pub trait ReadBytes: Read {
    #[inline]
    fn read_u8(&mut self) -> io::Result<u8> {
        let mut buf = [0; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u16<T: Byte>(&mut self) -> io::Result<u16> {
        let mut buf = [0; 2];
        self.read_exact(&mut buf)?;
        Ok(T::read_u16(&buf))
    }

    fn read_u32<T: Byte>(&mut self) -> io::Result<u32> {
        let mut buf = [0; 4];
        self.read_exact(&mut buf)?;
        Ok(T::read_u32(&buf))
    }
}

impl<R: Read + ?Sized> ReadBytes for R {}

pub trait Byte {
    fn read_u16(buf: &[u8]) -> u16;
    fn read_u32(buf: &[u8]) -> u32;
    fn write_u16(buf: &mut [u8], n: u16);
    fn write_u32(buf: &mut [u8], n: u32);
}

pub enum BigEndian {}

impl Byte for BigEndian {
    #[inline]
    fn read_u16(buf: &[u8]) -> u16 {
        ((buf[0] as u16) << 8) | (buf[1] as u16)
    }

    #[inline]
    fn read_u32(buf: &[u8]) -> u32 {
        ((buf[0] as u32) << 24) | ((buf[1] as u32) << 16) | ((buf[2] as u32) << 8) | (buf[3] as u32)
    }

    #[inline]
    fn write_u16(buf: &mut [u8], n: u16) {
        buf[0] = (n >> 8) as u8;
        buf[1] = n as u8;
    }

    #[inline]
    fn write_u32(buf: &mut [u8], n: u32) {
        buf[0] = (n >> 24) as u8;
        buf[1] = (n >> 16) as u8;
        buf[2] = (n >> 8) as u8;
        buf[3] = n as u8;
    }
}
