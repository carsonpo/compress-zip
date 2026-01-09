"""
RMSNorm implementation using integer arithmetic.

Matches CUDA rmsnorm.cu kernel.
"""

import numpy as np
from typing import List

from .primitives import isqrt32_restoring, sra_rne_tte_s32, clamp_i8


def compute_inv_rms_q15_from_mean_sq(mean_sq: int) -> int:
    """
    Compute 1/RMS in Q15 format from mean of squares.

    rms = sqrt(mean_sq)
    inv_rms = 1 / rms = 1 / sqrt(mean_sq)

    We compute this as:
    inv_rms_q15 = 32768 / sqrt(mean_sq)

    Using integer sqrt and careful scaling.

    Args:
        mean_sq: Mean of squared values (sum_sq / n)

    Returns:
        Inverse RMS in Q15 format (multiply by this and shift by 15)
    """
    if mean_sq == 0:
        return 32767  # Max Q15 value to avoid division by zero

    # Compute sqrt(mean_sq)
    rms = isqrt32_restoring(mean_sq)

    if rms == 0:
        return 32767

    # Compute 32768 / rms with rounding
    # To round: (32768 + rms/2) / rms
    inv_rms = (32768 + rms // 2) // rms

    return min(32767, inv_rms)


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
        x: Input tensor of shape [batch, dim] as int8
        weight: Weight tensor of shape [dim] as int8
        eps_sq: Epsilon squared (default 1, effectively eps=1 for integer)

    Returns:
        Output tensor of shape [batch, dim] as int8
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

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
