//! compress_zip - Deterministic neural compression verifier
//!
//! This crate provides bit-exact CPU implementations of neural compression
//! algorithms, designed to verify GPU-compressed data.
//!
//! All operations use deterministic integer arithmetic with explicit
//! ties-to-even rounding, matching the CUDA implementations exactly.

pub mod primitives;
pub mod lut;
pub mod rmsnorm;
pub mod reglu;
pub mod linear;
pub mod attention;
pub mod softmax_cdf;
pub mod arith_coder;
pub mod file_format;
pub mod safetensors;
pub mod tiktoken;

// Re-export commonly used types
pub use arith_coder::{ArithEncoder, ArithDecoder};
pub use file_format::{CZIPv1OuterHeader, CZIPv1InnerHeader, CompressedFile, Codec};
pub use primitives::*;
