"""
Softmax CDF builder for arithmetic coding.

Converts int32 logits to cumulative frequency table for arithmetic coder.
Matches CUDA softmax_cumfreq.cu exactly (softmax_cumfreq_i32_u32_kernel).
"""

import numpy as np
from typing import Tuple

from .primitives import argmax_deterministic
from .lut import get_exp2_lut

# Constants matching CUDA softmax_cumfreq.cu
NUM_STATE_BITS = 32
QUARTER_RANGE = 1 << (NUM_STATE_BITS - 2)  # 1 << 30 = 1073741824

# Exp coefficient matching CUDA (W_CLIP = 0.1, same as lm_head weight clip)
LOG2E = 1.4426950408889634
W_CLIP = 0.1  # Must match lm_head weight clip in model
COEF_D = (W_CLIP * LOG2E) / 64.0
COEF_Q24 = int(COEF_D * (1 << 24) + 0.5)  # ~37817


def compute_target_total(vocab_size: int) -> int:
    """
    Compute target total for frequency table.
    Matches CUDA: target_total = QUARTER_RANGE - vocab_size - 1024
    """
    target = QUARTER_RANGE - vocab_size - 1024
    # Safety check
    if target <= vocab_size:
        target = vocab_size + 1
    return target


def exp_q16_from_diff_acc(diff: int, exp_lut) -> int:
    """
    Compute exp2 for logit difference (int32 GEMM accumulator).
    Matches CUDA exp_q16_from_diff_acc exactly.

    diff = logit - max_logit (<= 0)
    Returns uint16 Q16 representation.
    """
    if diff >= 0:
        return 65535

    neg = -diff

    # t256 = round(neg * COEF_Q24 / 2^24)
    t256 = (neg * COEF_Q24 + (1 << 23)) >> 24

    ip = t256 >> 8  # integer part
    frac = t256 & 255  # fractional part

    if ip >= 31:
        return 1

    base = (1 << 16) >> ip

    # Get fractional multiplier from LUT
    frac_mul = int(exp_lut[frac])

    # Q16 * Q16 -> Q16 (rounded)
    out = (base * frac_mul + 0x8000) >> 16

    if out < 1:
        out = 1
    if out > 65535:
        out = 65535

    return out


def build_cumfreqs(
    logits: np.ndarray,
    vocab_size: int | None = None,
    exp_lut = None,
) -> Tuple[np.ndarray, int]:
    """
    Build cumulative frequency table from int32 logits.
    Matches CUDA softmax_cumfreq_i32_u32_kernel exactly.

    Algorithm:
    1. Find max logit and argmax (smallest index for ties)
    2. Compute exp weights using exp_q16_from_diff_acc
    3. Sum exp weights
    4. Allocate frequencies: freq = 1 + floor(exp * remaining / sum_exp)
    5. Add remainder to argmax
    6. Build inclusive cumsum (no leading 0)

    Args:
        logits: Int32 logits of shape [vocab_size]
        vocab_size: Optional vocab size (defaults to len(logits))
        exp_lut: Optional exp2 LUT (defaults to get_exp2_lut())

    Returns:
        Tuple of (cumfreqs array of shape [vocab_size], target_total)
        cumfreqs is inclusive prefix sum: cumfreqs[-1] = target_total
    """
    if vocab_size is None:
        vocab_size = len(logits)

    assert len(logits) >= vocab_size

    if exp_lut is None:
        exp_lut = get_exp2_lut()
    target_total = compute_target_total(vocab_size)

    # Pass 1: Find max logit
    max_val = int(np.max(logits[:vocab_size]))

    # Find argmax (smallest index for ties) - matches CUDA
    argmax_idx = argmax_deterministic(logits[:vocab_size].tolist())

    # Pass 2: Compute exp weights and sum
    exp_weights = []
    sum_exp = 0

    for i in range(vocab_size):
        diff = int(logits[i]) - max_val  # <= 0
        e = exp_q16_from_diff_acc(diff, exp_lut)
        exp_weights.append(e)
        sum_exp += e

    # Ensure sum_exp is at least 1 to avoid division by zero
    if sum_exp == 0:
        sum_exp = 1

    # Pass 3: Allocate frequencies
    # remaining = target_total - V
    # freq = 1 + floor(exp * remaining / sum_exp)
    remaining = target_total - vocab_size

    freqs = []
    freq_sum = 0

    for i in range(vocab_size):
        e = exp_weights[i]
        # e * remaining needs 64-bit (uint16 * uint32 = up to 2^46)
        add = (e * remaining) // sum_exp
        f = 1 + add
        freqs.append(f)
        freq_sum += f

    # Add remainder to argmax
    rem = target_total - freq_sum
    freqs[argmax_idx] += rem

    # Pass 4: Build inclusive cumsum (no leading 0)
    cumfreqs = np.zeros(vocab_size, dtype=np.uint32)
    running_sum = 0
    for i in range(vocab_size):
        running_sum += freqs[i]
        cumfreqs[i] = running_sum

    # Verify total
    assert cumfreqs[-1] == target_total, f"Total {cumfreqs[-1]} != target {target_total}"

    return cumfreqs, target_total


def build_cumfreqs_simple(
    logits: np.ndarray,
    vocab_size: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Simplified CDF builder using floating point softmax then quantizing.

    This is not bit-exact with the integer version but useful for testing.
    """
    if vocab_size is None:
        vocab_size = len(logits)

    target_total = compute_target_total(vocab_size)

    # Floating point softmax
    logits_f = logits[:vocab_size].astype(np.float64)
    logits_f = logits_f - np.max(logits_f)  # Numerical stability
    probs = np.exp(logits_f)
    probs = probs / np.sum(probs)

    # Quantize to frequencies
    freqs = np.round(probs * target_total).astype(np.int32)
    freqs = np.maximum(freqs, 1)  # Minimum freq of 1

    # Adjust to match target_total
    diff = target_total - np.sum(freqs)
    if diff != 0:
        max_idx = np.argmax(freqs)
        freqs[max_idx] += diff

    # Build cumulative frequencies
    cumfreqs = np.zeros(vocab_size + 1, dtype=np.int32)
    cumfreqs[1:] = np.cumsum(freqs)

    return cumfreqs, int(cumfreqs[vocab_size])


# Test cases
if __name__ == "__main__":
    # Test basic CDF building
    logits = np.array([1000, 2000, 3000, 4000], dtype=np.int32)
    cumfreqs, total = build_cumfreqs(logits)

    print(f"Logits: {logits}")
    print(f"Cumfreqs: {cumfreqs}")
    print(f"Total: {total}")

    # Verify properties
    assert cumfreqs[0] == 0
    assert cumfreqs[-1] == total
    assert all(cumfreqs[i] < cumfreqs[i+1] for i in range(len(cumfreqs)-1))

    # Test with uniform logits
    uniform_logits = np.array([1000, 1000, 1000, 1000], dtype=np.int32)
    cumfreqs_uniform, total_uniform = build_cumfreqs(uniform_logits)
    print(f"\nUniform logits: {uniform_logits}")
    print(f"Cumfreqs: {cumfreqs_uniform}")

    # Should be roughly uniform distribution
    freqs = np.diff(cumfreqs_uniform)
    print(f"Frequencies: {freqs}")

    # Test with larger vocab
    large_logits = np.random.randint(0, 10000, 1000, dtype=np.int32)
    cumfreqs_large, total_large = build_cumfreqs(large_logits, 1000)
    print(f"\nLarge vocab (1000): total = {total_large}")
    print(f"Min freq: {np.min(np.diff(cumfreqs_large))}")
    print(f"Max freq: {np.max(np.diff(cumfreqs_large))}")

    print("\nAll softmax CDF tests passed!")
