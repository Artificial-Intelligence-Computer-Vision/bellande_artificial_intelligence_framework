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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}

impl DataType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Bool => 1,
        }
    }

    pub fn is_floating_point(&self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    pub fn default() -> Self {
        DataType::Float32
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DataType::Float32 => write!(f, "float32"),
            DataType::Float64 => write!(f, "float64"),
            DataType::Int32 => write!(f, "int32"),
            DataType::Int64 => write!(f, "int64"),
            DataType::Bool => write!(f, "bool"),
        }
    }
}
