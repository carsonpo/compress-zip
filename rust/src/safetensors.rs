use half::f16;
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

// Constants from C++ port
const SAFETENSORS_MAX_DIM: usize = 8;
const SAFETENSORS_MAX_TENSORS: usize = 2048;
const SAFETENSORS_MAX_FILE_SIZE: u64 = 2u64 << 40; // 2 TiB
const SAFETENSORS_MAX_STRING_SIZE: usize = 2048;
const SAFETENSORS_MAX_METADATA_SIZE: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Int32,
    Int64,
    FP16,
    BF16,
    FP32,
    UInt8,
    Int8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::FP16 => 2,
            DType::BF16 => 2,
            DType::FP32 => 4,
            DType::UInt8 => 1,
            DType::Int8 => 1,
        }
    }
}

pub struct Tensor {
    data: Vec<f16>,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, dtype: DType, device: Device) -> Self {
        Self {
            data: vec![f16::from_f32(0.0); shape.iter().product()],
            shape,
            dtype,
            device,
        }
    }

    pub fn get_data_ptr(&self) -> *const f16 {
        self.data.as_ptr()
    }

    pub fn get_data_ptr_const(&self) -> *const f16 {
        self.data.as_ptr()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn from_vec(data: Vec<f16>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            data,
            shape,
            dtype,
            device: Device::CPU,
        }
    }

    pub fn as_ref(&self) -> &[f16] {
        &self.data
    }

    pub fn to_vec(&self) -> Vec<f16> {
        self.data.clone()
    }
}

use std::io;

// Modify the SafetensorsError to implement From for various error types
#[derive(Debug)]
pub enum SafetensorsError {
    IoError(io::Error),
    Utf8Error(std::str::Utf8Error),
    JsonError(serde_json::Error),
    Custom(String),
}

impl std::error::Error for SafetensorsError {}

impl std::fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SafetensorsError::IoError(e) => write!(f, "IO error: {}", e),
            SafetensorsError::Utf8Error(e) => write!(f, "UTF-8 error: {}", e),
            SafetensorsError::JsonError(e) => write!(f, "JSON error: {}", e),
            SafetensorsError::Custom(s) => write!(f, "Safetensors error: {}", s),
        }
    }
}

impl From<io::Error> for SafetensorsError {
    fn from(error: io::Error) -> Self {
        SafetensorsError::IoError(error)
    }
}

impl From<std::str::Utf8Error> for SafetensorsError {
    fn from(error: std::str::Utf8Error) -> Self {
        SafetensorsError::Utf8Error(error)
    }
}

impl From<serde_json::Error> for SafetensorsError {
    fn from(error: serde_json::Error) -> Self {
        SafetensorsError::JsonError(error)
    }
}

type Result<T> = std::result::Result<T, SafetensorsError>;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<i64>,
    #[serde(rename = "data_offsets")]
    data_offsets: [usize; 2],
}

// Helper functions
fn get_tensor_dtype(dtype_str: &str) -> Result<DType> {
    match dtype_str {
        "BOOL" | "I32" => Ok(DType::Int32),
        "I64" => Ok(DType::Int64),
        "F16" => Ok(DType::FP16),
        "BF16" => Ok(DType::BF16),
        "F32" => Ok(DType::FP32),
        "U8" => Ok(DType::UInt8),
        "I8" => Ok(DType::Int8),
        _ => Err(SafetensorsError::Custom(format!(
            "Unknown dtype: {}",
            dtype_str
        ))),
    }
}

fn get_safetensors_dtype(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::Int32 => Ok("I32"),
        DType::Int64 => Ok("I64"),
        DType::FP16 => Ok("F16"),
        DType::BF16 => Ok("BF16"),
        DType::FP32 => Ok("F32"),
        DType::UInt8 => Ok("U8"),
        DType::Int8 => Ok("I8"),
    }
}

fn validate_string_length(s: &String, context: &str) -> Result<()> {
    if s.len() > SAFETENSORS_MAX_STRING_SIZE {
        return Err(SafetensorsError::Custom(format!(
            "{} exceeds maximum allowed length",
            context
        )));
    }
    Ok(())
}

fn is_big_endian() -> bool {
    false
}

fn swap_bytes<T: Clone>(data: &mut [T]) {
    let size = std::mem::size_of::<T>();
    unsafe {
        let ptr = data.as_mut_ptr() as *mut u8;
        let len = data.len() * size;
        let slice = std::slice::from_raw_parts_mut(ptr, len);
        for chunk in slice.chunks_mut(size) {
            chunk.reverse();
        }
    }
}

pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor>> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();

    if file_size > SAFETENSORS_MAX_FILE_SIZE {
        return Err(SafetensorsError::Custom(
            "File size exceeds maximum allowed size".to_string(),
        ));
    }

    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // Read header size (first 8 bytes)
    let mut header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap());
    if is_big_endian() {
        header_size = header_size.swap_bytes();
    }

    if 8 + header_size > file_size {
        return Err(SafetensorsError::Custom("Invalid header size".to_string()));
    }

    // Parse header JSON
    let header_json = std::str::from_utf8(&mmap[8..8 + header_size as usize])?;
    // Remove `"__metadata__":{"format":"pt"},` from the JSON string if present
    let header_json = header_json.replace(r#""__metadata__":{"format":"pt"},"#, "");
    let tensor_infos: HashMap<String, TensorInfo> = serde_json::from_str(&header_json)?;

    if tensor_infos.len() > SAFETENSORS_MAX_TENSORS {
        return Err(SafetensorsError::Custom(
            "Number of tensors exceeds maximum allowed".to_string(),
        ));
    }

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let data_start = 8 + header_size as usize;

    for (name, info) in tensor_infos {
        validate_string_length(&name, "Tensor name")?;

        if info.shape.len() > SAFETENSORS_MAX_DIM {
            return Err(SafetensorsError::Custom(
                "Tensor dimension exceeds maximum allowed".to_string(),
            ));
        }

        let dtype = get_tensor_dtype(&info.dtype)?;
        let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();

        // Create CPU tensor and copy data
        let cpu_tensor = Tensor::new(shape.clone(), dtype, Device::CPU);
        let data_size = info.data_offsets[1] - info.data_offsets[0];

        unsafe {
            std::ptr::copy_nonoverlapping(
                mmap[data_start + info.data_offsets[0]..].as_ptr(),
                cpu_tensor.get_data_ptr() as *mut u8,
                data_size,
            );
        }

        // Move tensor to target device if needed
        let device_tensor = cpu_tensor;

        tensors.insert(name.to_string(), device_tensor);
    }

    Ok(tensors)
}

pub fn save_safetensors<P: AsRef<Path>>(
    tensors: &HashMap<String, Tensor>,
    path: P,
    metadata: Option<HashMap<String, String>>,
) -> Result<()> {
    if tensors.len() > SAFETENSORS_MAX_TENSORS {
        return Err(SafetensorsError::Custom(
            "Number of tensors exceeds maximum allowed".to_string(),
        ));
    }

    let mut header_json = serde_json::Map::new();
    let mut data_buffer = Vec::new();
    let mut current_offset = 0;

    // Add metadata if present
    if let Some(meta) = metadata {
        let mut metadata_map = serde_json::Map::new();
        for (key, value) in meta {
            validate_string_length(&key, "Metadata key")?;
            validate_string_length(&value, "Metadata value")?;
            metadata_map.insert(key, Value::String(value));
        }
        header_json.insert("__metadata__".to_string(), Value::Object(metadata_map));
    }

    // Process each tensor
    for (name, tensor) in tensors {
        validate_string_length(name, "Tensor name")?;

        let cpu_tensor = tensor;

        let tensor_info = TensorInfo {
            dtype: get_safetensors_dtype(cpu_tensor.dtype())?.to_string(),
            shape: cpu_tensor.shape().iter().map(|&x| x as i64).collect(),
            data_offsets: [
                current_offset,
                current_offset
                    + cpu_tensor.shape().iter().product::<usize>()
                        * cpu_tensor.dtype().size_in_bytes(),
            ],
        };

        // Handle endianness if needed
        if is_big_endian()
            && (cpu_tensor.dtype() == DType::FP16 || cpu_tensor.dtype() == DType::FP32)
        {
            let num_elements = tensor_info.data_offsets[1] - tensor_info.data_offsets[0];
            let mut data = unsafe {
                std::slice::from_raw_parts(cpu_tensor.get_data_ptr() as *const u8, num_elements)
                    .to_vec()
            };

            match cpu_tensor.dtype() {
                DType::FP16 => {
                    let data_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_mut_ptr() as *mut u16,
                            num_elements / 2,
                        )
                    };
                    swap_bytes(data_slice);
                }
                DType::FP32 => {
                    let data_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_mut_ptr() as *mut u32,
                            num_elements / 4,
                        )
                    };
                    swap_bytes(data_slice);
                }
                _ => unreachable!(),
            }
            data_buffer.extend_from_slice(&data);
        } else {
            // Copy tensor data to buffer
            let data = unsafe {
                std::slice::from_raw_parts(
                    cpu_tensor.get_data_ptr_const() as *const u8,
                    tensor_info.data_offsets[1] - tensor_info.data_offsets[0],
                )
            };
            data_buffer.extend_from_slice(data);
        }

        header_json.insert(name.clone(), serde_json::to_value(tensor_info.clone())?);
        current_offset = tensor_info.data_offsets[1];
    }

    let header_string = serde_json::to_string(&header_json)?;
    if header_string.len() > SAFETENSORS_MAX_METADATA_SIZE {
        return Err(SafetensorsError::Custom(
            "Metadata size exceeds maximum allowed size".to_string(),
        ));
    }

    let total_size = 8 + header_string.len() + data_buffer.len();
    if total_size as u64 > SAFETENSORS_MAX_FILE_SIZE {
        return Err(SafetensorsError::Custom(
            "Total file size exceeds maximum allowed size".to_string(),
        ));
    }

    let mut file = File::create(path)?;

    // Write header size
    let mut header_size = (header_string.len() as u64).to_le_bytes();
    if is_big_endian() {
        header_size.reverse();
    }
    file.write_all(&header_size)?;

    // Write header and data
    file.write_all(header_string.as_bytes())?;
    file.write_all(&data_buffer)?;

    Ok(())
}
