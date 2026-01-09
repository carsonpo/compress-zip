"""
Softmax CDF builder for arithmetic coding.

Converts int32 logits to cumulative frequency table for arithmetic coder.
"""

import numpy as np
from typing import Tuple

from .primitives import argmax_deterministic
from .lut import get_exp2_lut, exp_q16_from_neg_fixed


# Target total for frequency table (power of 2 for efficient coding)
TARGET_TOTAL_BITS = 16
TARGET_TOTAL = 1 << TARGET_TOTAL_BITS  # 65536


def compute_target_total(vocab_size: int) -> int:
    """
    Compute target total for frequency table.

    Must be large enough that each symbol gets at least frequency 1.
    Using 2^16 = 65536 works for vocab_size <= 65536.
    """
    # Ensure each symbol can have at least freq 1
    return max(TARGET_TOTAL, vocab_size)


def build_cumfreqs(
    logits: np.ndarray,
    vocab_size: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Build cumulative frequency table from int32 logits.

    Implements online softmax with integer arithmetic:
    1. Find max logit for numerical stability
    2. Compute exp(logit - max) for each symbol using exp2 LUT
    3. Normalize to target total
    4. Build cumulative frequencies

    Args:
        logits: Int32 logits of shape [vocab_size]
        vocab_size: Optional vocab size (defaults to len(logits))

    Returns:
        Tuple of (cumfreqs array of shape [vocab_size + 1], total)
        cumfreqs[0] = 0, cumfreqs[vocab_size] = total
    """
    if vocab_size is None:
        vocab_size = len(logits)

    assert len(logits) >= vocab_size

    exp_lut = get_exp2_lut()
    target_total = compute_target_total(vocab_size)

    # Find max logit
    max_logit = int(np.max(logits[:vocab_size]))

    # Compute unnormalized probabilities using exp2
    # We convert logit differences to exp2 input format

    # Scale factor: how many logit units per bit of exponent
    # This is tunable based on the expected range of logits
    # For int32 GEMM accumulators, logits can be large
    # Using a scale that maps reasonable differences to exp range
    LOGIT_SCALE_BITS = 8  # Shift logit diff right by this much

    probs = np.zeros(vocab_size, dtype=np.int64)
    total_prob = 0

    for i in range(vocab_size):
        diff = max_logit - int(logits[i])  # Always >= 0

        # Convert to Q8 exponent for exp2 lookup
        # Larger diff = smaller probability
        neg_exp_q8 = diff >> (LOGIT_SCALE_BITS - 8)  # Scale to Q8

        prob = exp_q16_from_neg_fixed(neg_exp_q8, exp_lut)
        probs[i] = prob
        total_prob += prob

    # Normalize to target_total
    if total_prob == 0:
        # All probabilities underflowed, give uniform distribution
        freq_per_symbol = target_total // vocab_size
        remainder = target_total - freq_per_symbol * vocab_size
        freqs = np.full(vocab_size, freq_per_symbol, dtype=np.int32)
        # Distribute remainder to first symbols
        for i in range(remainder):
            freqs[i] += 1
    else:
        # Scale probabilities to target_total
        freqs = np.zeros(vocab_size, dtype=np.int32)
        allocated = 0

        for i in range(vocab_size):
            # freq[i] = probs[i] * target_total / total_prob
            # Use 64-bit to avoid overflow
            freq = (probs[i] * target_total + total_prob // 2) // total_prob
            freq = max(1, int(freq))  # Ensure minimum freq of 1
            freqs[i] = freq
            allocated += freq

        # Adjust to exactly match target_total
        diff = target_total - allocated
        if diff != 0:
            # Find the symbol with highest probability and adjust it
            max_idx = argmax_deterministic([int(f) for f in freqs])
            freqs[max_idx] += diff

    # Build cumulative frequencies
    cumfreqs = np.zeros(vocab_size + 1, dtype=np.int32)
    cumfreqs[0] = 0
    for i in range(vocab_size):
        cumfreqs[i + 1] = cumfreqs[i] + freqs[i]

    total = int(cumfreqs[vocab_size])
    assert total == target_total, f"Total {total} != target {target_total}"

    return cumfreqs, total


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
