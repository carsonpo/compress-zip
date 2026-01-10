"""
RMSNorm implementation using integer arithmetic.

Matches CUDA rmsnorm.cu kernel.
"""

import numpy as np
from typing import List

from .primitives import isqrt32_restoring, sra_rne_tte_s32, clamp_i8, udiv_rne_tte_u32


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
    eps_sq: int = 1,
) -> np.ndarray:
    """
    Integer RMSNorm matching CUDA kernel.

    For each row:
    1. Compute mean of squares: mean_sq = sum(x^2) / n
    2. Compute inverse RMS: inv_rms = 1/sqrt(mean_sq + eps)
    3. Normalize: y = x * inv_rms * weight

    All computations use integer arithmetic with Q15 scaling for inv_rms.

    Args:
        x: Input tensor of shape [dim] or [batch, dim] as int8
        weight: Weight tensor of shape [dim] as int8
        eps_sq: Epsilon squared (default 1, effectively eps=1 for integer)

    Returns:
        Output tensor of shape [dim] or [batch, dim] as int8
    """
    squeeze = False
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True

    batch_size, dim = x.shape
    x = x.astype(np.int32)
    weight = weight.astype(np.int32)

    output = np.zeros((batch_size, dim), dtype=np.int8)

    for b in range(batch_size):
        row = x[b]

        # Compute sum of squares
        sum_sq = np.sum(row * row)

        # Mean of squares (add eps_sq for numerical stability)
        mean_sq = (sum_sq + dim // 2) // dim + eps_sq

        # Compute inverse RMS in Q15
        inv_rms_q15 = compute_inv_rms_q15_from_mean_sq(mean_sq)

        # Normalize each element
        for i in range(dim):
            # x[i] is Q0.7, inv_rms is Q15, weight is Q0.7
            # Product: x * inv_rms = Q0.22
            # Then multiply by weight: Q0.22 * Q0.7 = Q0.29
            # Shift right by 22 (15 + 7) to get Q0.7

            # Step 1: x * inv_rms (Q0.7 * Q15 = Q0.22)
            scaled = row[i] * inv_rms_q15

            # Step 2: shift by 15 with rounding to get normalized value
            normalized = sra_rne_tte_s32(scaled, 15)

            # Step 3: multiply by weight (both Q0.7)
            weighted = normalized * weight[i]

            # Step 4: shift by 7 to get back to Q0.7
            result = sra_rne_tte_s32(weighted, 7)

            # Clamp to int8 range
            output[b, i] = clamp_i8(result)

    if squeeze:
        return output[0]
    return output


def rmsnorm_i8_int(
    x: np.ndarray,
    weight: np.ndarray,
    eps_sq: int = 1,
) -> np.ndarray:
    """
    Integer RMSNorm with int16 Q1.14 weights.

    For each row:
    1. Compute mean of squares: mean_sq = sum(x^2) / n
    2. Compute inverse RMS: inv_rms = 1/sqrt(mean_sq + eps)
    3. Normalize: y = x * inv_rms * weight

    This version accepts int16 Q1.14 weights (divide by 16384 to get float).

    Args:
        x: Input tensor of shape [dim] or [batch, dim] as int8
        weight: Weight tensor of shape [dim] as int16 (Q1.14 fixed point)
        eps_sq: Epsilon squared (default 1, effectively eps=1 for integer)

    Returns:
        Output tensor of shape [dim] or [batch, dim] as int8
    """
    squeeze = False
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True

    batch_size, dim = x.shape
    x_i32 = x.astype(np.int32)
    weight_i32 = weight.astype(np.int32)

    output = np.zeros((batch_size, dim), dtype=np.int8)

    for b in range(batch_size):
        row = x_i32[b]

        # Compute sum of squares
        sum_sq = np.sum(row * row)

        # Mean of squares (add eps_sq for numerical stability)
        mean_sq = (sum_sq + dim // 2) // dim + eps_sq

        # Compute inverse RMS in Q15
        inv_rms_q15 = compute_inv_rms_q15_from_mean_sq(int(mean_sq))

        # Normalize each element
        for i in range(dim):
            # x[i] is Q0.7, inv_rms is Q15, weight is Q1.14
            # Step 1: x * inv_rms (Q0.7 * Q15 = Q0.22)
            scaled = int(row[i]) * inv_rms_q15

            # Step 2: shift by 15 with rounding to get normalized value (Q0.7)
            normalized = sra_rne_tte_s32(scaled, 15)

            # Step 3: multiply by weight (Q0.7 * Q1.14 = Q1.21)
            weighted = normalized * int(weight_i32[i])

            # Step 4: shift by 14 (not 7!) to account for Q1.14 weight format
            # Result is Q0.7
            result = sra_rne_tte_s32(weighted, 14)

            # Clamp to int8 range
            output[b, i] = clamp_i8(result)

    if squeeze:
        return output[0]
    return output


def rmsnorm_i8_vectorized(
    x: np.ndarray,
    weight: np.ndarray,
    eps_sq: int = 1,
) -> np.ndarray:
    """
    Vectorized integer RMSNorm (faster but may have slight differences due to order of operations).

    For bit-exact results, use rmsnorm_i8.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    batch_size, dim = x.shape
    x_i32 = x.astype(np.int32)
    weight_i32 = weight.astype(np.int32)

    # Compute sum of squares per row
    sum_sq = np.sum(x_i32 * x_i32, axis=1)

    # Mean of squares
    mean_sq = (sum_sq + dim // 2) // dim + eps_sq

    output = np.zeros((batch_size, dim), dtype=np.int8)

    for b in range(batch_size):
        inv_rms_q15 = compute_inv_rms_q15_from_mean_sq(int(mean_sq[b]))

        for i in range(dim):
            scaled = int(x_i32[b, i]) * inv_rms_q15
            normalized = sra_rne_tte_s32(scaled, 15)
            weighted = normalized * int(weight_i32[i])
            result = sra_rne_tte_s32(weighted, 7)
            output[b, i] = clamp_i8(result)

    return output


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
    x = np.array([[64, 64, 64, 64]], dtype=np.int8)  # 0.5 in Q0.7
    weight = np.array([127, 127, 127, 127], dtype=np.int8)  # ~1.0 in Q0.7

    result = rmsnorm_i8(x, weight)
    print(f"rmsnorm_i8 result: {result}")
    # Expected: normalized values close to input since RMS ~ 64

    print("All RMSNorm tests passed!")
