//! GQA (Grouped Query Attention) with integer RoPE and online softmax.
//!
//! Matches the CUDA implementation in attention.cu.

use crate::linear::linear_i8_to_i32;
use crate::lut::{exp_q16_from_neg_fixed_with_coef, Exp2LutQ16, RopeLut, ROPE_HALF_DIM, ROPE_HEAD_DIM};
use crate::primitives::{clamp_i8, div_rne_tte_s64_to_s32, sra_rne_tte_s32, sra_rne_tte_s64_to_s32};

/// Constants for attention
pub const HEAD_DIM: usize = 64;
pub const MAX_SEQ_LEN: usize = 64;
pub const TOKENS_PER_BATCH: usize = 16;

// Score to exp mapping coefficient for Q0.14 scores
// LOG2E = 1.4426950408889634
// ATTN_COEF = LOG2E / 64.0
// ATTN_COEF_Q24 = round(ATTN_COEF * 2^24)
pub const LOG2E: f64 = 1.4426950408889634;
pub const ATTN_COEF: f64 = LOG2E / 64.0;
pub const ATTN_COEF_Q24: u32 = ((ATTN_COEF * (1u64 << 24) as f64) + 0.5) as u32;

// Q and sqrt(d) shifts
pub const Q_SHIFT: u32 = 2;
pub const SQRT_HEAD_DIM_SHIFT: u32 = 3;

/// Apply RoPE rotation to a vector in-place.
///
/// Rotates pairs of dimensions using precomputed cos/sin values.
/// Input/output are in int8 Q0.7 format.
pub fn apply_rope_i8(x: &mut [i8], rope_lut: &RopeLut, pos: usize) {
    assert_eq!(x.len(), HEAD_DIM);

    for i in 0..ROPE_HALF_DIM {
        let xv = x[i] as i32;
        let yv = x[i + ROPE_HALF_DIM] as i32;

        let c = rope_lut.cos(pos, i) as i32;
        let s = rope_lut.sin(pos, i) as i32;

        // Rotate: [x', y'] = [x*c - y*s, x*s + y*c]
        let x_rot_raw = xv * c - yv * s;
        let y_rot_raw = xv * s + yv * c;

        // Shift right by 15 with ties-to-even (Q15 -> Q0)
        let x_rot = sra_rne_tte_s32(x_rot_raw, 15);
        let y_rot = sra_rne_tte_s32(y_rot_raw, 15);

        x[i] = clamp_i8(x_rot);
        x[i + ROPE_HALF_DIM] = clamp_i8(y_rot);
    }
}

/// Apply RoPE to Q with additional scaling for attention.
pub fn apply_rope_q_i8(x: &mut [i8], rope_lut: &RopeLut, pos: usize) {
    assert_eq!(x.len(), HEAD_DIM);

    const NET_SHIFT: i32 = (SQRT_HEAD_DIM_SHIFT as i32) - (Q_SHIFT as i32);

    for i in 0..ROPE_HALF_DIM {
        let xv = x[i] as i32;
        let yv = x[i + ROPE_HALF_DIM] as i32;

        let c = rope_lut.cos(pos, i) as i32;
        let s = rope_lut.sin(pos, i) as i32;

        let x_rot_raw = xv * c - yv * s;
        let y_rot_raw = xv * s + yv * c;

        let mut x_rot = sra_rne_tte_s32(x_rot_raw, 15);
        let mut y_rot = sra_rne_tte_s32(y_rot_raw, 15);

        // Apply net shift
        if NET_SHIFT > 0 {
            x_rot = sra_rne_tte_s32(x_rot, NET_SHIFT as u32);
            y_rot = sra_rne_tte_s32(y_rot, NET_SHIFT as u32);
        } else if NET_SHIFT < 0 {
            x_rot <<= (-NET_SHIFT) as u32;
            y_rot <<= (-NET_SHIFT) as u32;
        }

        x[i] = clamp_i8(x_rot);
        x[i + ROPE_HALF_DIM] = clamp_i8(y_rot);
    }
}

/// Compute Q*K dot product for int8 vectors.
///
/// Returns the raw dot product (no scaling yet).
pub fn qk_dot_i8(q: &[i8], k: &[i8]) -> i32 {
    assert_eq!(q.len(), HEAD_DIM);
    assert_eq!(k.len(), HEAD_DIM);

    let mut dot: i32 = 0;
    for i in 0..HEAD_DIM {
        dot += (q[i] as i32) * (k[i] as i32);
    }
    dot
}

/// KV Cache for a single layer.
#[derive(Clone)]
pub struct KVCache {
    /// K cache: [batch, max_seq_len, head_dim]
    pub k: Vec<i8>,
    /// V cache: [batch, max_seq_len, head_dim]
    pub v: Vec<i8>,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
}

impl KVCache {
    pub fn new(batch_size: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let size = batch_size * max_seq_len * head_dim;
        Self {
            k: vec![0i8; size],
            v: vec![0i8; size],
            batch_size,
            max_seq_len,
            head_dim,
        }
    }

    /// Get K for a specific batch and position.
    pub fn get_k(&self, batch: usize, pos: usize) -> &[i8] {
        let start = (batch * self.max_seq_len + pos) * self.head_dim;
        &self.k[start..start + self.head_dim]
    }

    /// Get V for a specific batch and position.
    pub fn get_v(&self, batch: usize, pos: usize) -> &[i8] {
        let start = (batch * self.max_seq_len + pos) * self.head_dim;
        &self.v[start..start + self.head_dim]
    }

    /// Set K for a specific batch and position.
    pub fn set_k(&mut self, batch: usize, pos: usize, k: &[i8]) {
        let start = (batch * self.max_seq_len + pos) * self.head_dim;
        self.k[start..start + self.head_dim].copy_from_slice(k);
    }

    /// Set V for a specific batch and position.
    pub fn set_v(&mut self, batch: usize, pos: usize, v: &[i8]) {
        let start = (batch * self.max_seq_len + pos) * self.head_dim;
        self.v[start..start + self.head_dim].copy_from_slice(v);
    }
}

/// GQA Attention forward pass for a single step.
///
/// This implements Multi-Query Attention (MQA) where n_kv_heads = 1.
///
/// Input:
/// - qkv: [B, D + 2*kv_dim] int8 Q0.7 (output of QKV projection)
/// - kv_cache: mutable KV cache
/// - step: current position (0-indexed)
/// - score_mul_q15: [n_heads] scaling factors in Q0.15
/// - rope_lut: precomputed RoPE LUT
/// - exp_lut: precomputed exp2 LUT
///
/// Output: [B, D] int8 Q0.7
pub fn gqa_attention_mqa_i8(
    qkv: &[i8],
    kv_cache: &mut KVCache,
    step: usize,
    n_heads: usize,
    score_mul_q15: &[i32],
    rope_lut: &RopeLut,
    exp_lut: &Exp2LutQ16,
) -> Vec<i8> {
    assert!(n_heads == 4 || n_heads == 6 || n_heads == 8);
    let d = n_heads * HEAD_DIM;
    let kv_dim = HEAD_DIM; // MQA: single KV head

    let b = qkv.len() / (d + 2 * kv_dim);
    let tokens = step + 1;

    let mut out = vec![0i8; b * d];

    for batch in 0..b {
        let qkv_start = batch * (d + 2 * kv_dim);
        let k_offset = d;
        let v_offset = d + kv_dim;

        // Extract and rotate K, store to cache
        let mut k_rot = [0i8; HEAD_DIM];
        k_rot.copy_from_slice(&qkv[qkv_start + k_offset..qkv_start + k_offset + HEAD_DIM]);
        apply_rope_i8(&mut k_rot, rope_lut, step);
        kv_cache.set_k(batch, step, &k_rot);

        // Store V to cache (no rotation)
        let v_slice = &qkv[qkv_start + v_offset..qkv_start + v_offset + HEAD_DIM];
        kv_cache.set_v(batch, step, v_slice);

        // Process each head
        for h in 0..n_heads {
            // Extract and rotate Q for this head
            let q_offset = h * HEAD_DIM;
            let mut q_rot = [0i8; HEAD_DIM];
            q_rot.copy_from_slice(&qkv[qkv_start + q_offset..qkv_start + q_offset + HEAD_DIM]);
            apply_rope_q_i8(&mut q_rot, rope_lut, step);

            // Online softmax state
            let mut m: i32 = i32::MIN; // running max score
            let mut s: u64 = 0; // running sum of exp weights
            let mut vacc = [0i64; HEAD_DIM]; // weighted V accumulator

            // Compute attention over all tokens
            for t in 0..tokens {
                let k = kv_cache.get_k(batch, t);

                // Compute Q*K dot product
                let dot = qk_dot_i8(&q_rot, k);

                // Scale: dot_unshift = round(dot >> Q_SHIFT)
                let dot_unshift = sra_rne_tte_s32(dot, Q_SHIFT);

                // Apply score multiplier: score = round(dot_unshift * mul >> 15)
                let mul = score_mul_q15[h];
                let prod = (dot_unshift as i64) * (mul as i64);
                let score = sra_rne_tte_s64_to_s32(prod, 15);

                // Update online softmax
                if score > m {
                    // New max found - need to rescale
                    if m != i32::MIN {
                        let diff = m - score;
                        let scale_old = exp_q16_from_neg_fixed_with_coef(diff, exp_lut, ATTN_COEF_Q24);
                        // Rescale S
                        s = (s * scale_old as u64 + 0x8000) >> 16;
                        // Rescale vacc - must match CUDA exactly
                        // CUDA: prod += (prod >= 0) ? (1LL << 15) : -(1LL << 15); vacc = prod >> 16;
                        for v in vacc.iter_mut() {
                            let prod = (*v) * (scale_old as i64);
                            // Match CUDA's rounding: add/subtract half then arithmetic shift
                            let adjusted = if prod >= 0 {
                                prod + (1i64 << 15)
                            } else {
                                prod - (1i64 << 15)
                            };
                            *v = adjusted >> 16;
                        }
                    }
                    m = score;
                }

                // Compute exp weight for this token
                let diff = score - m;
                let w = exp_q16_from_neg_fixed_with_coef(diff, exp_lut, ATTN_COEF_Q24);
                s += w as u64;

                // Accumulate weighted V
                let v = kv_cache.get_v(batch, t);
                for i in 0..HEAD_DIM {
                    vacc[i] += (w as i64) * (v[i] as i64);
                }
            }

            // Normalize output
            let denom = if s == 0 { 1 } else { s };
            let out_start = batch * d + h * HEAD_DIM;
            for i in 0..HEAD_DIM {
                let y = div_rne_tte_s64_to_s32(vacc[i], denom);
                out[out_start + i] = clamp_i8(y);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rope_identity_at_pos0() {
        let rope_lut = RopeLut::new(10000.0);
        let mut x = vec![64i8; HEAD_DIM];

        // At position 0, cos=1, sin=0, so rotation should be identity
        apply_rope_i8(&mut x, &rope_lut, 0);

        // Values should be close to original (within rounding)
        for &v in &x {
            assert!((v as i32 - 64).abs() <= 1);
        }
    }

    #[test]
    fn test_qk_dot() {
        let q = vec![10i8; HEAD_DIM];
        let k = vec![10i8; HEAD_DIM];

        let dot = qk_dot_i8(&q, &k);
        assert_eq!(dot, 10 * 10 * HEAD_DIM as i32);
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new(2, MAX_SEQ_LEN, HEAD_DIM);

        let k = vec![42i8; HEAD_DIM];
        cache.set_k(0, 5, &k);

        let retrieved = cache.get_k(0, 5);
        assert_eq!(retrieved, k.as_slice());
    }

    #[test]
    fn test_attn_coef() {
        // Verify the coefficient is computed correctly
        let expected = ((LOG2E / 64.0) * (1u64 << 24) as f64 + 0.5) as u32;
        assert_eq!(ATTN_COEF_Q24, expected);
    }
}
