//! Int8 linear layer (GEMM).
//!
//! Computes int8 x int8 matrix multiplication, outputting int32 accumulators.

/// Int8 linear layer (matrix multiplication).
///
/// Input: x [B, in_dim] - int8
/// Weight: w [out_dim, in_dim] - int8 (row-major)
/// Output: [B, out_dim] - int32 accumulators
///
/// Computation: out[b, o] = sum_i(x[b, i] * w[o, i])
pub fn linear_i8_to_i32(x: &[i8], w: &[i8], b: usize, in_dim: usize, out_dim: usize) -> Vec<i32> {
    let mut out = vec![0i32; b * out_dim];

    for batch in 0..b {
        let x_row = &x[batch * in_dim..(batch + 1) * in_dim];

        for o in 0..out_dim {
            let w_row = &w[o * in_dim..(o + 1) * in_dim];

            let mut acc: i32 = 0;
            for i in 0..in_dim {
                acc += (x_row[i] as i32) * (w_row[i] as i32);
            }

            out[batch * out_dim + o] = acc;
        }
    }

    out
}

/// Int8 linear layer with int8 output (clamped).
///
/// This is a convenience function that clamps the int32 accumulators to int8.
/// Note: This loses precision and should only be used where the model expects it.
pub fn linear_i8_to_i8(x: &[i8], w: &[i8], b: usize, in_dim: usize, out_dim: usize) -> Vec<i8> {
    let out_i32 = linear_i8_to_i32(x, w, b, in_dim, out_dim);
    out_i32
        .iter()
        .map(|&v| v.clamp(-127, 127) as i8)
        .collect()
}

/// Int8 linear layer with scaled int8 output.
///
/// Scales the accumulator by `scale` (in Q0.15 format) before clamping.
/// out = clamp(round(acc * scale / 2^15), -127, 127)
pub fn linear_i8_scaled(
    x: &[i8],
    w: &[i8],
    b: usize,
    in_dim: usize,
    out_dim: usize,
    scale_q15: i32,
) -> Vec<i8> {
    use crate::primitives::sra_rne_tte_s64_to_s32;

    let out_i32 = linear_i8_to_i32(x, w, b, in_dim, out_dim);
    out_i32
        .iter()
        .map(|&acc| {
            let prod = (acc as i64) * (scale_q15 as i64);
            let scaled = sra_rne_tte_s64_to_s32(prod, 15);
            scaled.clamp(-127, 127) as i8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_simple() {
        // 1x2 input, 2x2 weight -> 1x2 output
        let x = vec![1i8, 2];
        let w = vec![
            1i8, 0, // row 0
            0, 1, // row 1
        ];

        let out = linear_i8_to_i32(&x, &w, 1, 2, 2);
        assert_eq!(out, vec![1, 2]); // identity-ish
    }

    #[test]
    fn test_linear_dot_product() {
        // [1, 2, 3] dot [1, 1, 1] = 6
        let x = vec![1i8, 2, 3];
        let w = vec![1i8, 1, 1];

        let out = linear_i8_to_i32(&x, &w, 1, 3, 1);
        assert_eq!(out, vec![6]);
    }

    #[test]
    fn test_linear_batch() {
        // 2 batches, 2 inputs, 1 output
        let x = vec![
            1i8, 2, // batch 0
            3, 4, // batch 1
        ];
        let w = vec![1i8, 1]; // sum

        let out = linear_i8_to_i32(&x, &w, 2, 2, 1);
        assert_eq!(out, vec![3, 7]);
    }

    #[test]
    fn test_linear_negative() {
        let x = vec![-10i8, 10];
        let w = vec![1i8, 1];

        let out = linear_i8_to_i32(&x, &w, 1, 2, 1);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn test_linear_accumulator_range() {
        // Max accumulator: 127 * 127 * dim
        // For dim=64, max = 127 * 127 * 64 = 1,032,256
        let x = vec![127i8; 64];
        let w = vec![127i8; 64];

        let out = linear_i8_to_i32(&x, &w, 1, 64, 1);
        assert_eq!(out[0], 127 * 127 * 64);
    }
}
