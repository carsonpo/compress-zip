//! Core deterministic primitives for bit-exact neural compression.
//!
//! All operations use explicit integer arithmetic with ties-to-even rounding,
//! matching the CUDA implementations exactly.

/// Clamp a value to the symmetric int8 range [-127, 127].
/// Matches CUDA: max(-127, min(127, x))
#[inline]
pub fn clamp_i8(x: i32) -> i8 {
    x.clamp(-127, 127) as i8
}

/// Signed 32-bit shift-right with ties-to-even rounding.
///
/// Returns `round(v / 2^sh)` where ties are rounded to the nearest even number.
/// This matches the CUDA `sra_round_ties_to_even_s32` exactly.
#[inline]
pub fn sra_rne_tte_s32(v: i32, sh: u32) -> i32 {
    if sh == 0 {
        return v;
    }

    // Get sign and absolute value
    let sign = v >> 31; // -1 if negative, 0 if positive
    let av = ((v ^ sign) - sign) as u32; // absolute value without overflow

    let mask = (1u32 << sh) - 1;
    let r = av & mask; // remainder (discarded bits)
    let mut q = av >> sh; // quotient

    let half = 1u32 << (sh - 1);
    if r > half {
        q += 1;
    } else if r == half {
        // Ties-to-even: round up if quotient is odd
        q += q & 1;
    }

    // Restore sign
    let out = q as i32;
    (out ^ sign) - sign
}

/// 64-bit to 32-bit shift-right with ties-to-even rounding.
///
/// Returns `round(v / 2^sh)` clamped to i32 range.
/// This matches the CUDA `sra_round_ties_to_even_s64_to_s32` exactly.
#[inline]
pub fn sra_rne_tte_s64_to_s32(v: i64, sh: u32) -> i32 {
    if sh == 0 {
        return v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }

    // Get sign and absolute value
    let sign = v >> 63; // -1 if negative, 0 if positive
    let av = ((v ^ sign) - sign) as u64; // absolute value

    let mask = (1u64 << sh) - 1;
    let r = av & mask;
    let mut q = av >> sh;

    let half = 1u64 << (sh - 1);
    if r > half {
        q += 1;
    } else if r == half {
        q += q & 1; // ties-to-even
    }

    // Restore sign and clamp to i32
    let out = q as i64;
    let signed_out = (out ^ sign) - sign;
    signed_out.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Unsigned 32-bit division with ties-to-even rounding.
///
/// Returns `round(n / d)` where ties are rounded to nearest even.
/// This matches the CUDA `udiv_round_ties_to_even_u32` from rmsnorm.cu.
#[inline]
pub fn udiv_rne_tte_u32(n: u32, d: u32) -> u32 {
    if d == 0 {
        return 0; // Avoid division by zero
    }

    let q = n / d;
    let r = n - q * d; // remainder

    let twice_r = r << 1;
    if twice_r > d {
        q + 1
    } else if twice_r == d {
        // Ties-to-even: round up if quotient is odd
        q + (q & 1)
    } else {
        q
    }
}

/// Signed 32-bit division with ties-to-even rounding.
///
/// Returns `round(num / denom)` where ties are rounded to nearest even.
#[inline]
pub fn div_rne_tte_s32(num: i32, denom: i32) -> i32 {
    if denom == 0 {
        return 0;
    }

    let sign = if (num < 0) != (denom < 0) { -1 } else { 1 };
    let abs_num = num.unsigned_abs();
    let abs_denom = denom.unsigned_abs();

    let q = abs_num / abs_denom;
    let r = abs_num - q * abs_denom;

    let twice_r = r << 1;
    let rounded_q = if twice_r > abs_denom {
        q + 1
    } else if twice_r == abs_denom {
        q + (q & 1)
    } else {
        q
    };

    (rounded_q as i32) * sign
}

/// Signed 64-bit to 32-bit division with ties-to-even rounding.
///
/// Returns `round(num / denom)` clamped to i32 range.
/// This is used in attention output normalization.
#[inline]
pub fn div_rne_tte_s64_to_s32(num: i64, denom: u64) -> i32 {
    if denom == 0 {
        return 0;
    }

    let sign: i64 = if num < 0 { -1 } else { 1 };
    let abs_num = num.unsigned_abs();

    let q = abs_num / denom;
    let r = abs_num - q * denom;

    let twice_r = r << 1;
    let rounded_q = if twice_r > denom {
        q + 1
    } else if twice_r == denom {
        q + (q & 1)
    } else {
        q
    };

    // Restore sign and clamp
    let out = (rounded_q as i64) * sign;
    out.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Signed 32-bit divided by unsigned 64-bit with ties-to-even rounding.
///
/// Returns `round(num / denom)` as i32.
/// Optimized version for attention output normalization where numerator fits in i32.
/// This allows the compiler to use faster 32-bit operations where possible.
#[inline]
pub fn div_rne_tte_s32_by_u64(num: i32, denom: u64) -> i32 {
    if denom == 0 {
        return 0;
    }

    let sign: i32 = if num < 0 { -1 } else { 1 };
    let abs_num = num.unsigned_abs() as u64;

    let q = abs_num / denom;
    let r = abs_num - q * denom;

    let twice_r = r << 1;
    let rounded_q = if twice_r > denom {
        q + 1
    } else if twice_r == denom {
        q + (q & 1)
    } else {
        q
    };

    // Restore sign (result always fits in i32 since num was i32 and we divided)
    (rounded_q as i32) * sign
}

/// Integer square root using the restoring algorithm.
///
/// Returns `floor(sqrt(x))`. This is a pure integer operation with no
/// floating point, matching the CUDA `isqrt32_restoring` exactly.
#[inline]
pub fn isqrt32_restoring(x: u32) -> u32 {
    let mut op = x;
    let mut res = 0u32;
    let mut one = 1u32 << 30;

    // Find the highest power of 4 <= x
    while one > op {
        one >>= 2;
    }

    while one != 0 {
        if op >= res + one {
            op -= res + one;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }

    res
}

/// Quantize f32 to int8 Q0.7 with ties-to-even rounding.
///
/// Q0.7 format: real_value = int8_value / 128
/// Range: [-127/128, 127/128] ~ [-0.992, 0.992]
#[inline]
pub fn quant_f32_to_i8_q07(x: f32, clip: f32) -> i8 {
    let clamped = x.clamp(-clip, clip);
    // Use round-to-nearest-even (Rust's default for f32::round is ties-away,
    // so we use a manual implementation)
    let scaled = clamped * 128.0;
    let q = round_ties_to_even_f32(scaled) as i32;
    q.clamp(-127, 127) as i8
}

/// Dequantize int8 Q0.7 to f32.
#[inline]
pub fn dequant_i8_q07_to_f32(x: i8) -> f32 {
    (x as f32) * (1.0 / 128.0)
}

/// Round f32 to nearest integer with ties-to-even.
/// This matches IEEE-754 default rounding mode.
#[inline]
pub fn round_ties_to_even_f32(x: f32) -> f32 {
    // Rust's f32::round() rounds ties away from zero, not to even.
    // We need to implement ties-to-even manually.
    let rounded = x.round();
    let diff = x - x.floor();

    // Check if we're exactly at 0.5 (a tie)
    if diff == 0.5 {
        // Round to even: if floor is odd, use ceil; if even, use floor
        let floor_val = x.floor();
        if (floor_val as i32) & 1 == 0 {
            floor_val
        } else {
            x.ceil()
        }
    } else {
        rounded
    }
}

/// Round i64 to nearest integer with ties-to-even (for host LUT generation).
#[inline]
pub fn round_ties_to_even_host(x: f64) -> i64 {
    // Use the same logic as above but for f64 -> i64
    let rounded = x.round();
    let floor_val = x.floor();
    let diff = x - floor_val;

    if (diff - 0.5).abs() < f64::EPSILON {
        if (floor_val as i64) & 1 == 0 {
            floor_val as i64
        } else {
            x.ceil() as i64
        }
    } else {
        rounded as i64
    }
}

/// Compute argmax with deterministic tie-breaking (smallest index wins).
pub fn argmax_deterministic(arr: &[i32]) -> usize {
    if arr.is_empty() {
        return 0;
    }

    let mut max_idx = 0;
    let mut max_val = arr[0];

    for (i, &val) in arr.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
        // Note: ties are NOT updated, so smallest index wins
    }

    max_idx
}

/// Constants for Q0.7 format
pub const Q07_SCALE: f32 = 128.0;
pub const Q07_INV_SCALE: f32 = 1.0 / 128.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sra_rne_tte_s32_exact() {
        // Exact divisions (no remainder)
        assert_eq!(sra_rne_tte_s32(100, 1), 50);
        assert_eq!(sra_rne_tte_s32(64, 2), 16);
        assert_eq!(sra_rne_tte_s32(-100, 1), -50);
    }

    #[test]
    fn test_sra_rne_tte_s32_ties_to_even() {
        // 3 >> 1 = 1.5 -> round to 2 (even)
        assert_eq!(sra_rne_tte_s32(3, 1), 2);
        // 5 >> 1 = 2.5 -> round to 2 (even)
        assert_eq!(sra_rne_tte_s32(5, 1), 2);
        // 7 >> 1 = 3.5 -> round to 4 (even)
        assert_eq!(sra_rne_tte_s32(7, 1), 4);
        // Negative
        assert_eq!(sra_rne_tte_s32(-3, 1), -2);
        assert_eq!(sra_rne_tte_s32(-5, 1), -2);
        assert_eq!(sra_rne_tte_s32(-7, 1), -4);
    }

    #[test]
    fn test_sra_rne_tte_s32_round_up() {
        // 5 >> 2 = 1.25 -> round to 1
        assert_eq!(sra_rne_tte_s32(5, 2), 1);
        // 7 >> 2 = 1.75 -> round to 2
        assert_eq!(sra_rne_tte_s32(7, 2), 2);
    }

    #[test]
    fn test_isqrt32_restoring() {
        assert_eq!(isqrt32_restoring(0), 0);
        assert_eq!(isqrt32_restoring(1), 1);
        assert_eq!(isqrt32_restoring(4), 2);
        assert_eq!(isqrt32_restoring(9), 3);
        assert_eq!(isqrt32_restoring(15), 3);
        assert_eq!(isqrt32_restoring(16), 4);
        assert_eq!(isqrt32_restoring(65536), 256);
        assert_eq!(isqrt32_restoring(0xFFFFFFFF), 65535);
    }

    #[test]
    fn test_udiv_rne_tte_u32() {
        // Exact divisions
        assert_eq!(udiv_rne_tte_u32(10, 2), 5);
        assert_eq!(udiv_rne_tte_u32(12, 4), 3);

        // Ties-to-even
        // 3 / 2 = 1.5 -> round to 2 (even)
        assert_eq!(udiv_rne_tte_u32(3, 2), 2);
        // 5 / 2 = 2.5 -> round to 2 (even)
        assert_eq!(udiv_rne_tte_u32(5, 2), 2);
        // 7 / 2 = 3.5 -> round to 4 (even)
        assert_eq!(udiv_rne_tte_u32(7, 2), 4);

        // Round up (> 0.5)
        assert_eq!(udiv_rne_tte_u32(7, 4), 2); // 1.75 -> 2
    }

    #[test]
    fn test_clamp_i8() {
        // Clamps to symmetric range [-127, 127] for quantization
        assert_eq!(clamp_i8(0), 0);
        assert_eq!(clamp_i8(127), 127);
        assert_eq!(clamp_i8(128), 127);
        assert_eq!(clamp_i8(-127), -127);
        assert_eq!(clamp_i8(-128), -127);  // Symmetric range uses -127
        assert_eq!(clamp_i8(-129), -127);  // Underflow clamps to -127
        assert_eq!(clamp_i8(1000), 127);
        assert_eq!(clamp_i8(-1000), -127);
    }

    #[test]
    fn test_quant_dequant_roundtrip() {
        // Test that small values roundtrip correctly
        for i in -100..=100 {
            let f = i as f32 / 128.0;
            let q = quant_f32_to_i8_q07(f, 1.0);
            // Allow ±1 error due to quantization
            assert!((q as i32 - i).abs() <= 1, "Failed for i={}: got {}", i, q);
        }
    }

    #[test]
    fn test_argmax_deterministic() {
        // Simple case
        assert_eq!(argmax_deterministic(&[1, 3, 2]), 1);

        // Ties: smallest index wins
        assert_eq!(argmax_deterministic(&[3, 3, 1]), 0);
        assert_eq!(argmax_deterministic(&[1, 3, 3]), 1);
    }
}
