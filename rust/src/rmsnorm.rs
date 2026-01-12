//! Integer RMSNorm implementation.
//!
//! Matches the CUDA implementation in rmsnorm.cu exactly.

use crate::primitives::{clamp_i8, isqrt32_restoring, sra_rne_tte_s32, udiv_rne_tte_u32};

/// Compute inverse RMS in Q15 format from mean square value.
/// Matches CUDA compute_inv_rms_q15_from_mean_sq exactly.
///
/// Uses scaled approach for better precision:
/// sqrt_scaled = sqrt(mean_sq << 16) = sqrt(mean_sq) * 256
/// inv_rms_q15 = 8388608 / sqrt_scaled = 32768 / sqrt(mean_sq)
#[inline]
pub fn compute_inv_rms_q15_from_mean_sq(mean_sq: i32) -> i32 {
    // Clamp to at least 1 to avoid division by zero
    let ms = if mean_sq > 0 { mean_sq as u32 } else { 1u32 };

    // Scale up by 2^16 for better precision
    let x_scaled = ms << 16;

    // Compute sqrt(mean_sq * 2^16) = sqrt(mean_sq) * 256
    let sqrt_scaled = isqrt32_restoring(x_scaled);

    // Compute 8388608 / sqrt_scaled = 32768 * 256 / (sqrt(mean_sq) * 256) = 32768 / sqrt(mean_sq)
    // Using ties-to-even rounding
    let inv = udiv_rne_tte_u32(8388608, sqrt_scaled);

    // Clamp to valid range [1, 32768]
    inv.clamp(1, 32768) as i32
}

/// Apply RMSNorm to int8 input.
///
/// Input: x_i8 [B, D] - int8 Q0.7
/// Weight: gamma_i8 [D] - int8 Q0.7
/// Output: [B, D] - int32 in Q0.7 units
///
/// The output is int32 because it will be clamped to int8 by the caller
/// (matching the GPU behavior where RMSNorm returns int32 and Python clamps).
pub fn rmsnorm_i8(x: &[i8], weight: &[i8], d: usize, eps_scaled: i32) -> Vec<i32> {
    let b = x.len() / d;
    let mut out = vec![0i32; x.len()];

    for batch in 0..b {
        let row_start = batch * d;
        let row = &x[row_start..row_start + d];

        // Compute sum of squares
        let sum_sq: u32 = row.iter().map(|&v| (v as i32 * v as i32) as u32).sum();

        // mean_sq = floor(sum_sq / D) + eps_scaled
        let mean_sq = (sum_sq / d as u32) as i32 + eps_scaled;
        let mean_sq = mean_sq.max(1);

        // Compute inverse RMS in Q15
        let inv_rms_q15 = compute_inv_rms_q15_from_mean_sq(mean_sq);

        // Apply: y = round_to_even(x * inv_rms_q15 * w / 2^15)
        for i in 0..d {
            let xv = row[i] as i32;
            let wv = weight[i] as i32;
            let tmp = (xv * wv) * inv_rms_q15; // fits in i32: max |xv*wv*inv| < 127*127*32768 ≈ 528M
            let y = sra_rne_tte_s32(tmp, 15);
            out[row_start + i] = y;
        }
    }

    out
}

/// Apply RMSNorm and return int8 output (clamped).
pub fn rmsnorm_i8_to_i8(x: &[i8], weight: &[i8], d: usize, eps_scaled: i32) -> Vec<i8> {
    let out_i32 = rmsnorm_i8(x, weight, d, eps_scaled);
    out_i32.iter().map(|&y| clamp_i8(y)).collect()
}

/// Compute eps_scaled from epsilon value.
///
/// eps_scaled = round(eps * 16384) in mean(x_i8^2) domain
pub fn compute_eps_scaled(eps: f64) -> i32 {
    let eps_scaled = (eps * 16384.0).round() as i32;
    eps_scaled.max(1)
}

/// Apply RMSNorm with int16 Q1.14 weights (czip-model-v1 format).
///
/// Input: x_i8 [D] - int8 Q0.7
/// Weight: gamma_i16 [D] - int16 Q1.14 (divide by 16384 to get float)
/// Output: [D] - int8
///
/// Matches Python rmsnorm_i8_int function exactly.
pub fn rmsnorm_i8_int(x: &[i8], weight: &[i16], d: usize) -> Vec<i8> {
    let eps_sq = 1i32;

    // Compute sum of squares (max = 127^2 * 64 = 1,032,256 fits in i32)
    let sum_sq: i32 = x.iter().map(|&v| (v as i32) * (v as i32)).sum();

    // Mean of squares (add eps_sq for numerical stability)
    let mean_sq = (sum_sq + (d as i32 / 2)) / d as i32 + eps_sq;
    let mean_sq = mean_sq.max(1);

    // Compute inverse RMS in Q15
    let inv_rms_q15 = compute_inv_rms_q15_from_mean_sq(mean_sq);

    // Normalize each element
    let mut out = vec![0i8; d];
    for i in 0..d {
        let xv = x[i] as i32;
        let wv = weight[i] as i32;

        // Step 1: x * inv_rms (Q0.7 * Q15 = Q0.22)
        let scaled = xv * inv_rms_q15;

        // Step 2: shift by 15 with rounding to get normalized value (Q0.7)
        let normalized = sra_rne_tte_s32(scaled, 15);

        // Step 3: multiply by weight (Q0.7 * Q1.14 = Q1.21)
        let weighted = normalized * wv;

        // Step 4: shift by 14 (not 7!) to account for Q1.14 weight format
        // Result is Q0.7
        let result = sra_rne_tte_s32(weighted, 14);

        // Clamp to int8 range
        out[i] = clamp_i8(result);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_inv_rms_q15() {
        // For mean_sq = 1: sqrt_scaled = sqrt(1 << 16) = 256, inv = 8388608/256 = 32768
        assert_eq!(compute_inv_rms_q15_from_mean_sq(1), 32768);

        // For mean_sq = 4: sqrt_scaled = sqrt(4 << 16) = 512, inv = 8388608/512 = 16384
        assert_eq!(compute_inv_rms_q15_from_mean_sq(4), 16384);

        // For mean_sq = 16: sqrt_scaled = sqrt(16 << 16) = 1024, inv = 8388608/1024 = 8192
        assert_eq!(compute_inv_rms_q15_from_mean_sq(16), 8192);

        // For mean_sq = 16384: sqrt_scaled = sqrt(16384 << 16) = 32768, inv = 8388608/32768 = 256
        assert_eq!(compute_inv_rms_q15_from_mean_sq(16384), 256);
    }

    #[test]
    fn test_rmsnorm_simple() {
        // Simple test: all zeros should give all zeros
        let x = vec![0i8; 4];
        let weight = vec![127i8; 4];
        let eps_scaled = compute_eps_scaled(1e-5);
        let out = rmsnorm_i8(&x, &weight, 4, eps_scaled);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_rmsnorm_unit() {
        // Test with unit values
        let x = vec![64i8, 64, 64, 64]; // ~0.5 in Q0.7
        let weight = vec![127i8; 4]; // ~1.0 in Q0.7
        let eps_scaled = compute_eps_scaled(1e-5);
        let out = rmsnorm_i8(&x, &weight, 4, eps_scaled);

        // Output should be normalized, all values should be similar
        for &v in &out {
            assert!(v.abs() <= 127);
        }
    }

    #[test]
    fn test_eps_scaled() {
        let eps = 1e-5;
        let eps_scaled = compute_eps_scaled(eps);
        // round(1e-5 * 16384) ≈ 0, but clamped to 1
        assert!(eps_scaled >= 1);
    }
}
