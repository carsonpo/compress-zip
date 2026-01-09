//! ReGLU (ReLU-Gated Linear Unit) activation.
//!
//! Matches the CUDA implementation in swiglu.cu.

use crate::primitives::{clamp_i8, sra_rne_tte_s32};

/// Apply ReGLU activation to int8 input.
///
/// Input: x [B, 2*d_ff] - int8 Q0.7
///   - First half [0..d_ff): gate values
///   - Second half [d_ff..2*d_ff): value to gate
///
/// Output: [B, d_ff] - int8 Q0.7
///
/// Computation: out = ReLU(gate) * value
///   - ReLU: max(0, gate)
///   - Product: Q0.7 * Q0.7 = Q0.14
///   - Shift right by 7 with ties-to-even to get Q0.7
///   - Clamp to [-127, 127]
pub fn reglu_i8(x: &[i8], d_ff: usize) -> Vec<i8> {
    let b = x.len() / (2 * d_ff);
    let mut out = Vec::with_capacity(b * d_ff);

    for batch in 0..b {
        let row_start = batch * 2 * d_ff;

        for i in 0..d_ff {
            let gate = x[row_start + i] as i32;
            let value = x[row_start + d_ff + i] as i32;

            // ReLU on gate
            let gate_relu = gate.max(0);

            // Product in Q0.14 (7+7=14 fractional bits)
            let prod = gate_relu * value;

            // Shift right by 7 with ties-to-even to get back to Q0.7
            let y = sra_rne_tte_s32(prod, 7);

            // Clamp to int8 range
            out.push(clamp_i8(y));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reglu_relu_zeros_negative_gate() {
        // Negative gate should be zeroed by ReLU
        let x = vec![-64i8, 100, 50, 100]; // gate=-64, value=100
        let out = reglu_i8(&x, 1);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 0); // -64 becomes 0 via ReLU
    }

    #[test]
    fn test_reglu_positive_gate() {
        // Positive gate: 64 * 64 = 4096 in Q0.14, >> 7 = 32 in Q0.7
        let x = vec![64i8, 64];
        let out = reglu_i8(&x, 1);
        assert_eq!(out.len(), 1);
        // 64 * 64 = 4096, 4096 >> 7 = 32
        assert_eq!(out[0], 32);
    }

    #[test]
    fn test_reglu_batch() {
        // Two batches
        let x = vec![
            64i8, 64, // batch 0: gate=64, value=64
            32, 64, // batch 1: gate=32, value=64
        ];
        let out = reglu_i8(&x, 1);
        assert_eq!(out.len(), 2);
        // Batch 0: 64*64 >> 7 = 32
        assert_eq!(out[0], 32);
        // Batch 1: 32*64 >> 7 = 16
        assert_eq!(out[1], 16);
    }

    #[test]
    fn test_reglu_ties_to_even() {
        // Test ties-to-even rounding
        // Need to find values where product >> 7 hits a tie
        // 32 * 2 = 64, 64 >> 7 = 0.5 -> should round to 0 (even)
        let x = vec![32i8, 2];
        let out = reglu_i8(&x, 1);
        assert_eq!(out[0], 0); // 64 >> 7 = 0.5 -> 0 (even)

        // 96 * 2 = 192, 192 >> 7 = 1.5 -> should round to 2 (even)
        let x = vec![96i8, 2];
        let out = reglu_i8(&x, 1);
        assert_eq!(out[0], 2); // 192 >> 7 = 1.5 -> 2 (even)
    }
}
