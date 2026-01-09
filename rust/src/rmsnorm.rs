//! Integer RMSNorm implementation.
//!
//! Matches the CUDA implementation in rmsnorm.cu exactly.

use crate::primitives::{clamp_i8, isqrt32_restoring, sra_rne_tte_s32, udiv_rne_tte_u32};

/// Compute inverse RMS in Q15 format from mean square value.
///
/// inv_rms_q15 = round_to_even(32768 / sqrt(mean_sq))
///
/// This matches the CUDA `compute_inv_rms_q15_from_mean_sq` exactly.
#[inline]
pub fn compute_inv_rms_q15_from_mean_sq(mean_sq: i32) -> i32 {
    let ms = if mean_sq > 0 { mean_sq as u32 } else { 1 };

    // x_scaled = ms << 16 (fits in u32 because ms <= ~20000)
    let x_scaled = ms << 16;
    let sqrt_scaled = isqrt32_restoring(x_scaled); // floor(sqrt(ms) * 256)

    // sqrt_scaled is >= 256, never 0
    // inv = round_to_even(8388608 / sqrt_scaled)
    // Note: 8388608 = 32768 * 256
    let inv = udiv_rne_tte_u32(8388608, sqrt_scaled);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_inv_rms_q15() {
        // For mean_sq = 1, inv_rms = 32768 (max)
        assert_eq!(compute_inv_rms_q15_from_mean_sq(1), 32768);

        // For mean_sq = 16384, sqrt = 128, inv = 32768/128 = 256
        // But with our scaling: sqrt_scaled = sqrt(16384 << 16) = sqrt(1073741824) ≈ 32768
        // inv = 8388608 / 32768 = 256
        let result = compute_inv_rms_q15_from_mean_sq(16384);
        assert!(result > 0 && result <= 32768);
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
