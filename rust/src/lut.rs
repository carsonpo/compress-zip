//! Look-up tables for deterministic operations.
//!
//! Contains precomputed tables for:
//! - Exp2 fractional approximation (for softmax/attention)
//! - RoPE cos/sin values (for rotary position embeddings)

use crate::primitives::round_ties_to_even_host;

/// Constants for exp2 LUT
pub const EXP_FRAC_BITS: usize = 8;
pub const EXP_FRAC_SIZE: usize = 1 << EXP_FRAC_BITS; // 256

/// Constants for RoPE LUT
pub const ROPE_MAX_SEQ_LEN: usize = 64;
pub const ROPE_HALF_DIM: usize = 32;
pub const ROPE_HEAD_DIM: usize = 64;

/// Exp2 fractional LUT in Q16 format.
///
/// `exp2_neg_frac_q16[i] = round(2^(-i/256) * 65536)`
///
/// This is used for integer approximation of `exp2(-x)` where x has
/// 8 fractional bits.
#[derive(Clone)]
pub struct Exp2LutQ16 {
    pub table: [u16; EXP_FRAC_SIZE],
}

impl Default for Exp2LutQ16 {
    fn default() -> Self {
        Self::new()
    }
}

impl Exp2LutQ16 {
    /// Generate the exp2 fractional LUT.
    pub fn new() -> Self {
        let mut table = [0u16; EXP_FRAC_SIZE];

        for i in 0..EXP_FRAC_SIZE {
            let frac = i as f64 / EXP_FRAC_SIZE as f64;
            let v = 2.0_f64.powf(-frac) * 65536.0;
            let q = round_ties_to_even_host(v);
            table[i] = q.clamp(1, 65535) as u16;
        }

        Self { table }
    }

    /// Look up exp2(-frac) where frac is in [0, 1) with 8 fractional bits.
    #[inline]
    pub fn lookup(&self, frac_bits: u8) -> u16 {
        self.table[frac_bits as usize]
    }
}

/// RoPE (Rotary Position Embedding) LUT in Q1.15 format.
///
/// Contains precomputed cos/sin values for each position and dimension.
#[derive(Clone)]
pub struct RopeLut {
    /// cos values: [pos][dim] in Q1.15 format
    pub cos_q15: [[i16; ROPE_HALF_DIM]; ROPE_MAX_SEQ_LEN],
    /// sin values: [pos][dim] in Q1.15 format
    pub sin_q15: [[i16; ROPE_HALF_DIM]; ROPE_MAX_SEQ_LEN],
}

impl RopeLut {
    /// Generate RoPE LUT for the given theta value.
    pub fn new(theta: f64) -> Self {
        let mut cos_q15 = [[0i16; ROPE_HALF_DIM]; ROPE_MAX_SEQ_LEN];
        let mut sin_q15 = [[0i16; ROPE_HALF_DIM]; ROPE_MAX_SEQ_LEN];

        for pos in 0..ROPE_MAX_SEQ_LEN {
            for i in 0..ROPE_HALF_DIM {
                let inv_freq = theta.powf(-2.0 * (i as f64) / (ROPE_HEAD_DIM as f64));
                let angle = (pos as f64) * inv_freq;

                let c = angle.cos();
                let s = angle.sin();

                // Convert to Q1.15: multiply by 32768 and round
                let c_q15 = round_ties_to_even_host(c * 32768.0);
                let s_q15 = round_ties_to_even_host(s * 32768.0);

                cos_q15[pos][i] = c_q15.clamp(-32767, 32767) as i16;
                sin_q15[pos][i] = s_q15.clamp(-32767, 32767) as i16;
            }
        }

        Self { cos_q15, sin_q15 }
    }

    /// Get cos value for given position and dimension.
    #[inline]
    pub fn cos(&self, pos: usize, dim: usize) -> i16 {
        self.cos_q15[pos][dim]
    }

    /// Get sin value for given position and dimension.
    #[inline]
    pub fn sin(&self, pos: usize, dim: usize) -> i16 {
        self.sin_q15[pos][dim]
    }
}

impl Default for RopeLut {
    fn default() -> Self {
        Self::new(10000.0)
    }
}

/// Compute exp2(-x) approximation using integer LUT.
///
/// Input: `neg_x` is a negative value in some fixed-point format
/// Output: Q16 approximation of exp2(neg_x / scale)
///
/// This is used in attention softmax where we compute exp(score - max_score).
#[inline]
pub fn exp_q16_from_neg_fixed(neg_x: i32, exp_lut: &Exp2LutQ16, coef_q24: u32) -> u16 {
    if neg_x >= 0 {
        return 65535; // exp(0) = 1.0 in Q16
    }

    let neg = (-neg_x) as u32;

    // t256 = round(neg * coef_q24 / 2^24)
    let t256 = ((neg as u64 * coef_q24 as u64 + (1u64 << 23)) >> 24) as u32;

    let ip = t256 >> 8; // integer part
    let frac = (t256 & 255) as u8; // fractional part (8 bits)

    if ip >= 31 {
        return 1; // Underflow to minimum
    }

    let base = (1u32 << 16) >> ip;
    let frac_mul = exp_lut.lookup(frac) as u32;
    let out = (base * frac_mul + 0x8000) >> 16;

    out.clamp(1, 65535) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp2_lut_generation() {
        let lut = Exp2LutQ16::new();

        // exp2(0) = 1.0, so table[0] should be 65536 (clamped to 65535)
        assert_eq!(lut.table[0], 65535);

        // exp2(-0.5) ≈ 0.707, so table[128] ≈ 46341
        let expected_half = (2.0_f64.powf(-0.5) * 65536.0).round() as u16;
        assert!((lut.table[128] as i32 - expected_half as i32).abs() <= 1);

        // All values should be in valid range
        for &v in &lut.table {
            assert!(v >= 1);
        }
    }

    #[test]
    fn test_rope_lut_generation() {
        let lut = RopeLut::new(10000.0);

        // At position 0, angle = 0 for all dims
        // cos(0) = 1.0 = 32767 in Q15, sin(0) = 0
        assert_eq!(lut.cos(0, 0), 32767);
        assert_eq!(lut.sin(0, 0), 0);

        // Values should be in valid Q15 range
        for pos in 0..ROPE_MAX_SEQ_LEN {
            for dim in 0..ROPE_HALF_DIM {
                assert!(lut.cos(pos, dim) >= -32767 && lut.cos(pos, dim) <= 32767);
                assert!(lut.sin(pos, dim) >= -32767 && lut.sin(pos, dim) <= 32767);
            }
        }
    }
}
