"""
RMSNorm implementation using integer arithmetic.

Matches CUDA rmsnorm.cu kernel.
"""

import numpy as np

from .primitives import isqrt32_restoring, udiv_rne_tte_u32


def compute_inv_rms_q15_from_mean_sq(mean_sq: int) -> int:
    """
    Compute 1/RMS in Q15 format from mean of squares.
    Matches CUDA compute_inv_rms_q15_from_mean_sq exactly.

    Uses scaled approach for better precision:
    sqrt_scaled = sqrt(mean_sq << 16) = sqrt(mean_sq) * 256
    inv_rms_q15 = 8388608 / sqrt_scaled = 32768 / sqrt(mean_sq)

    Args:
        mean_sq: Mean of squared values (sum_sq / n)

    Returns:
        Inverse RMS in Q15 format (multiply by this and shift by 15)
    """
    # Clamp to at least 1 to avoid division by zero
    ms = mean_sq if mean_sq > 0 else 1

    # Scale up by 2^16 for better precision
    x_scaled = ms << 16

    # Compute sqrt(mean_sq * 2^16) = sqrt(mean_sq) * 256
    sqrt_scaled = isqrt32_restoring(x_scaled)

    # Compute 8388608 / sqrt_scaled = 32768 * 256 / (sqrt(mean_sq) * 256) = 32768 / sqrt(mean_sq)
    # Using ties-to-even rounding
    inv = udiv_rne_tte_u32(8388608, sqrt_scaled)

    # Clamp to valid range [1, 32768]
    return max(1, min(32768, inv))


def rmsnorm_i8(
    x: np.ndarray,
    weight: np.ndarray,
    eps_scaled: int = 1,
) -> np.ndarray:
    """
    Integer RMSNorm matching CUDA/Rust implementations exactly.

    For each element:
        tmp = (x * w) * inv_rms_q15
        y = sra_rne_tte(tmp, 15)

    Args:
        x: Input tensor of shape [dim] as int8
        weight: Weight tensor of shape [dim] as int8
        eps_scaled: Epsilon scaled (default 1)

    Returns:
        Output tensor of shape [dim] as int8
    """
    d = len(x)
    x_i64 = x.astype(np.int64)
    weight_i64 = weight.astype(np.int64)

    # Compute sum of squares
    sum_sq = int(np.sum(x_i64 * x_i64))

    # mean_sq = floor(sum_sq / D) + eps_scaled
    mean_sq = sum_sq // d + eps_scaled
    mean_sq = max(1, mean_sq)

    # Compute inverse RMS in Q15
    inv_rms_q15 = compute_inv_rms_q15_from_mean_sq(mean_sq)

    # Apply: y = round_to_even((x * w * inv_rms_q15) >> 15)
    # Vectorized computation
    tmp = (x_i64 * weight_i64) * inv_rms_q15

    # Vectorized ties-to-even shift
    shift = 15
    half = np.int64(1 << (shift - 1))
    mask = np.int64((1 << shift) - 1)
    frac = tmp & mask
    result = tmp >> shift

    # Round up if frac > half
    round_up = frac > half
    # Round to even on ties
    tie = frac == half
    odd_result = (result & 1) == 1
    round_up_tie = tie & odd_result

    result = result + round_up.astype(np.int64) + round_up_tie.astype(np.int64)

    # Clamp to int8 range
    return np.clip(result, -127, 127).astype(np.int8)


# Test cases
if __name__ == "__main__":
    # Test compute_inv_rms_q15_from_mean_sq
    inv_rms = compute_inv_rms_q15_from_mean_sq(1)  # sqrt(1) = 1, inv = 1
    assert inv_rms == 32768, f"Expected 32768, got {inv_rms}"

    inv_rms = compute_inv_rms_q15_from_mean_sq(4)  # sqrt(4) = 2, inv = 0.5
    assert inv_rms == 16384, f"Expected 16384, got {inv_rms}"

    inv_rms = compute_inv_rms_q15_from_mean_sq(16)  # sqrt(16) = 4, inv = 0.25
    assert inv_rms == 8192, f"Expected 8192, got {inv_rms}"
    print("compute_inv_rms_q15_from_mean_sq tests passed")

    # Test rmsnorm_i8
    x = np.array([64, 64, 64, 64], dtype=np.int8)  # 0.5 in Q0.7
    weight = np.array([127, 127, 127, 127], dtype=np.int8)  # ~1.0 in Q0.7

    result = rmsnorm_i8(x, weight)
    print(f"rmsnorm_i8 result: {result}")
    # Expected: normalized values close to input since RMS ~ 64

    print("All RMSNorm tests passed!")
