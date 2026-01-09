//! Softmax + CDF builder from int32 logits.
//!
//! Converts raw GEMM accumulators (int32) to cumulative frequencies
//! for arithmetic coding.

use crate::lut::{Exp2LutQ16, EXP_FRAC_SIZE};
use crate::primitives::{argmax_deterministic, round_ties_to_even_host};

/// Arithmetic coder constants
pub const NUM_STATE_BITS: u32 = 32;
pub const STATE_MASK: u64 = (1u64 << NUM_STATE_BITS) - 1;
pub const HALF_RANGE: u64 = 1u64 << (NUM_STATE_BITS - 1);
pub const QUARTER_RANGE: u64 = 1u64 << (NUM_STATE_BITS - 2);

/// Compute the coefficient for logits exp mapping.
///
/// coef_q24 = round(scale_factor * 2^24)
///
/// For raw GEMM accumulators with lm_head weight clip, scale_factor should be:
/// log2(e) * lm_head_w_clip / 16384
pub fn compute_coef_q24_for_acc(lm_head_w_clip: f64) -> u32 {
    const LOG2E: f64 = 1.4426950408889634;
    let scale = LOG2E * lm_head_w_clip / 16384.0;
    round_ties_to_even_host(scale * (1u64 << 24) as f64).clamp(1, u32::MAX as i64) as u32
}

/// Compute target_total for frequency allocation.
///
/// Must be < QUARTER_RANGE and leave room for all frequencies being at least 1.
pub fn compute_target_total(vocab_size: usize) -> u32 {
    // target_total = QUARTER_RANGE - vocab_size - 1024
    let target = (QUARTER_RANGE as i64) - (vocab_size as i64) - 1024;
    target.max(vocab_size as i64 + 1) as u32
}

/// Compute exp weight from logit difference using integer LUT.
///
/// Input: diff = logit - max_logit (always <= 0)
/// Output: Q16 approximation of exp(diff * scale)
///
/// This matches the CUDA exp_q16_from_diff_acc function.
#[inline]
pub fn exp_q16_from_diff(diff: i32, exp_lut: &Exp2LutQ16, coef_q24: u32) -> u16 {
    if diff >= 0 {
        return 65535; // exp(0) = 1.0 in Q16
    }

    let neg = (-diff) as u32;

    // t256 = round(neg * coef_q24 / 2^24)
    let t256 = ((neg as u64 * coef_q24 as u64 + (1u64 << 23)) >> 24) as u32;

    let ip = t256 >> 8; // integer part (number of bit shifts)
    let frac = (t256 & 255) as u8; // fractional part (LUT index)

    if ip >= 31 {
        return 1; // Underflow to minimum
    }

    let base = (1u32 << 16) >> ip;
    let frac_mul = exp_lut.lookup(frac) as u32;
    let out = (base * frac_mul + 0x8000) >> 16;

    out.clamp(1, 65535) as u16
}

/// Build cumulative frequencies from int32 logits.
///
/// Input: logits [vocab_size] - raw int32 GEMM accumulators
/// Output: cumfreqs [vocab_size] - cumulative frequencies (inclusive prefix sum)
///
/// The last element of cumfreqs equals target_total.
pub fn build_cumfreqs(
    logits: &[i32],
    exp_lut: &Exp2LutQ16,
    coef_q24: u32,
    target_total: u32,
) -> Vec<u32> {
    let v = logits.len();

    // Step 1: Find max logit and argmax (smallest index for ties)
    let max_val = *logits.iter().max().unwrap_or(&0);
    let argmax_idx = argmax_deterministic(logits);

    // Step 2: Compute exp weights
    let mut exp_weights = Vec::with_capacity(v);
    let mut sum_exp: u64 = 0;

    for &logit in logits {
        let diff = logit - max_val;
        let e = exp_q16_from_diff(diff, exp_lut, coef_q24);
        exp_weights.push(e);
        sum_exp += e as u64;
    }

    // Step 3: Allocate frequencies
    // Each token gets at least 1, plus proportional allocation
    let remaining = target_total as u64 - v as u64;
    let mut freqs = Vec::with_capacity(v);
    let mut freq_sum: u64 = 0;

    for &e in &exp_weights {
        // freq = 1 + floor(e * remaining / sum_exp)
        let extra = (e as u64 * remaining) / sum_exp;
        let freq = 1 + extra as u32;
        freqs.push(freq);
        freq_sum += freq as u64;
    }

    // Step 4: Add remainder to argmax
    let rem = target_total as u64 - freq_sum;
    freqs[argmax_idx] += rem as u32;

    // Step 5: Compute cumulative sum (inclusive prefix sum)
    let mut cumfreqs = Vec::with_capacity(v);
    let mut running_sum: u32 = 0;

    for freq in freqs {
        running_sum += freq;
        cumfreqs.push(running_sum);
    }

    // Verify: last element should equal target_total
    debug_assert_eq!(cumfreqs[v - 1], target_total);

    cumfreqs
}

/// Build cumulative frequencies for a batch of logits.
pub fn build_cumfreqs_batch(
    logits: &[i32], // [B, V] flattened
    vocab_size: usize,
    exp_lut: &Exp2LutQ16,
    coef_q24: u32,
    target_total: u32,
) -> Vec<Vec<u32>> {
    let b = logits.len() / vocab_size;
    let mut result = Vec::with_capacity(b);

    for batch in 0..b {
        let start = batch * vocab_size;
        let row = &logits[start..start + vocab_size];
        result.push(build_cumfreqs(row, exp_lut, coef_q24, target_total));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exp_lut() -> Exp2LutQ16 {
        Exp2LutQ16::new()
    }

    #[test]
    fn test_exp_q16_from_diff() {
        let lut = make_exp_lut();
        // Use a more realistic lm_head_w_clip value to get meaningful exp values
        let coef = compute_coef_q24_for_acc(128.0);

        // diff = 0 should give max (65535)
        assert_eq!(exp_q16_from_diff(0, &lut, coef), 65535);

        // Negative diff should give smaller values
        let e1 = exp_q16_from_diff(-100, &lut, coef);
        let e2 = exp_q16_from_diff(-200, &lut, coef);
        assert!(e1 > e2, "e1={} should be > e2={}", e1, e2);
        assert!(e1 < 65535, "e1={} should be < 65535", e1);
        assert!(e2 >= 1, "e2={} should be >= 1", e2);
    }

    #[test]
    fn test_build_cumfreqs_sum() {
        let lut = make_exp_lut();
        let coef = compute_coef_q24_for_acc(0.1);
        let target_total = compute_target_total(16);

        let logits = vec![100i32, 50, 25, 10, 5, 0, -5, -10, -25, -50, -100, -200, -300, -400, -500, -1000];

        let cumfreqs = build_cumfreqs(&logits, &lut, coef, target_total);

        // Last element should equal target_total
        assert_eq!(cumfreqs[15], target_total);

        // Should be monotonically increasing
        for i in 1..cumfreqs.len() {
            assert!(cumfreqs[i] > cumfreqs[i - 1]);
        }
    }

    #[test]
    fn test_build_cumfreqs_min_freq() {
        let lut = make_exp_lut();
        let coef = compute_coef_q24_for_acc(0.1);
        let target_total = compute_target_total(4);

        // One very high logit, others very low
        let logits = vec![10000i32, -10000, -10000, -10000];

        let cumfreqs = build_cumfreqs(&logits, &lut, coef, target_total);

        // All tokens should have at least freq=1
        assert!(cumfreqs[0] >= 1);
        for i in 1..4 {
            assert!(cumfreqs[i] - cumfreqs[i - 1] >= 1);
        }
    }

    #[test]
    fn test_target_total() {
        let target = compute_target_total(1024);
        assert!(target < QUARTER_RANGE as u32);
        assert!(target > 1024);
    }

    #[test]
    fn test_argmax_gets_remainder() {
        let lut = make_exp_lut();
        let coef = compute_coef_q24_for_acc(0.1);
        let target_total = compute_target_total(4);

        // Identical logits - first one should get remainder (smallest index)
        let logits = vec![100i32, 100, 100, 100];

        let cumfreqs = build_cumfreqs(&logits, &lut, coef, target_total);

        // First token (argmax due to tie-breaking) should have highest freq
        let freq0 = cumfreqs[0];
        let freq1 = cumfreqs[1] - cumfreqs[0];
        assert!(freq0 >= freq1);
    }
}
