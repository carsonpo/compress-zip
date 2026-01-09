"""
Deterministic integer primitives matching CUDA helpers.cu.

All functions use ties-to-even rounding for determinism.
"""

import numpy as np
from typing import Sequence


def sra_rne_tte_s32(x: int, shift: int) -> int:
    """
    Shift right arithmetic with round-to-nearest, ties-to-even.

    Matches CUDA sra_round_ties_to_even_s32 in helpers.cu:1571-1592.

    Args:
        x: Signed 32-bit integer value
        shift: Number of bits to shift right (0-31)

    Returns:
        Rounded shifted value as signed 32-bit integer
    """
    if shift == 0:
        return x
    if shift >= 32:
        return 0

    # Half of the divisor for rounding
    half = 1 << (shift - 1)

    # The fractional part that's being shifted out
    frac = x & ((1 << shift) - 1)

    # Arithmetic shift right
    result = x >> shift

    # Round ties to even
    if frac > half:
        result += 1
    elif frac == half:
        # Tie: round to even (if result is odd, round up)
        if result & 1:
            result += 1

    return result


def sra_rne_tte_s64_to_s32(x: int, shift: int) -> int:
    """
    Shift right arithmetic from 64-bit to 32-bit with round-to-nearest, ties-to-even.

    Args:
        x: Signed 64-bit integer value
        shift: Number of bits to shift right

    Returns:
        Rounded shifted value clamped to signed 32-bit integer
    """
    if shift == 0:
        result = x
    elif shift >= 64:
        return 0
    else:
        half = 1 << (shift - 1)
        frac = x & ((1 << shift) - 1)
        result = x >> shift

        if frac > half:
            result += 1
        elif frac == half:
            if result & 1:
                result += 1

    # Clamp to i32 range
    return max(-2147483648, min(2147483647, result))


def udiv_rne_tte_u32(numerator: int, divisor: int) -> int:
    """
    Unsigned division with round-to-nearest, ties-to-even.

    Matches CUDA udiv_round_ties_to_even_u32 in rmsnorm.cu:81-95.

    Args:
        numerator: Unsigned 32-bit numerator
        divisor: Unsigned 32-bit divisor (must be > 0)

    Returns:
        Rounded quotient
    """
    if divisor == 0:
        raise ValueError("Division by zero")

    quotient = numerator // divisor
    remainder = numerator % divisor
    half = divisor // 2

    # Round ties to even
    if remainder > half:
        quotient += 1
    elif remainder == half:
        # For even divisors, exact tie
        if divisor % 2 == 0:
            if quotient & 1:
                quotient += 1
        else:
            # For odd divisors, remainder == half means round up
            quotient += 1

    return quotient


def div_rne_tte_s32(numerator: int, divisor: int) -> int:
    """
    Signed division with round-to-nearest, ties-to-even.

    Args:
        numerator: Signed 32-bit numerator
        divisor: Signed 32-bit divisor (must be != 0)

    Returns:
        Rounded quotient
    """
    if divisor == 0:
        raise ValueError("Division by zero")

    # Handle signs
    neg = (numerator < 0) != (divisor < 0)
    num_abs = abs(numerator)
    div_abs = abs(divisor)

    quotient = num_abs // div_abs
    remainder = num_abs % div_abs
    half = div_abs // 2

    # Round ties to even
    if remainder > half:
        quotient += 1
    elif remainder == half:
        if div_abs % 2 == 0:
            if quotient & 1:
                quotient += 1
        else:
            quotient += 1

    return -quotient if neg else quotient


def isqrt32_restoring(x: int) -> int:
    """
    Integer square root using restoring algorithm.

    Matches CUDA isqrt32_restoring in rmsnorm.cu:99-121.

    Args:
        x: Unsigned 32-bit integer

    Returns:
        Floor of square root of x
    """
    if x == 0:
        return 0

    # Find highest set bit position
    bit = 1 << 30  # Start with bit at position 30

    result = 0
    while bit > x:
        bit >>= 2

    while bit != 0:
        if x >= result + bit:
            x -= result + bit
            result = (result >> 1) + bit
        else:
            result >>= 1
        bit >>= 2

    return result


def clamp_i8(x: int) -> int:
    """
    Clamp value to signed 8-bit range [-128, 127].

    Args:
        x: Integer value

    Returns:
        Value clamped to i8 range
    """
    return max(-128, min(127, x))


def argmax_deterministic(values: Sequence[int]) -> int:
    """
    Deterministic argmax: returns index of maximum value.
    On ties, returns the lowest index (deterministic).

    Args:
        values: Sequence of integer values

    Returns:
        Index of maximum value
    """
    if not values:
        raise ValueError("Empty sequence")

    max_val = values[0]
    max_idx = 0

    for i, v in enumerate(values[1:], 1):
        if v > max_val:
            max_val = v
            max_idx = i

    return max_idx


# Numpy vectorized versions for batch operations

def sra_rne_tte_s32_np(x: np.ndarray, shift: int) -> np.ndarray:
    """Vectorized shift right arithmetic with ties-to-even rounding."""
    if shift == 0:
        return x.copy()
    if shift >= 32:
        return np.zeros_like(x)

    half = 1 << (shift - 1)
    mask = (1 << shift) - 1

    frac = x & mask
    result = x >> shift

    # Round up if frac > half
    round_up = frac > half
    # Round to even on ties
    tie = frac == half
    odd_result = (result & 1) == 1
    round_up_tie = tie & odd_result

    result = result + round_up.astype(result.dtype) + round_up_tie.astype(result.dtype)
    return result


def clamp_i8_np(x: np.ndarray) -> np.ndarray:
    """Vectorized clamp to i8 range."""
    return np.clip(x, -128, 127).astype(np.int8)


# Test cases
if __name__ == "__main__":
    # Test sra_rne_tte_s32
    assert sra_rne_tte_s32(7, 1) == 4  # 3.5 -> 4 (tie to even)
    assert sra_rne_tte_s32(5, 1) == 2  # 2.5 -> 2 (tie to even)
    assert sra_rne_tte_s32(6, 1) == 3  # 3.0 -> 3
    assert sra_rne_tte_s32(3, 1) == 2  # 1.5 -> 2 (tie to even)
    assert sra_rne_tte_s32(-7, 1) == -4  # -3.5 -> -4
    print("sra_rne_tte_s32 tests passed")

    # Test isqrt32_restoring
    assert isqrt32_restoring(0) == 0
    assert isqrt32_restoring(1) == 1
    assert isqrt32_restoring(4) == 2
    assert isqrt32_restoring(9) == 3
    assert isqrt32_restoring(10) == 3
    assert isqrt32_restoring(15) == 3
    assert isqrt32_restoring(16) == 4
    assert isqrt32_restoring(1000000) == 1000
    print("isqrt32_restoring tests passed")

    # Test udiv_rne_tte_u32
    assert udiv_rne_tte_u32(7, 2) == 4  # 3.5 -> 4 (tie to even)
    assert udiv_rne_tte_u32(5, 2) == 2  # 2.5 -> 2 (tie to even)
    assert udiv_rne_tte_u32(10, 4) == 2  # 2.5 -> 2 (tie to even)
    assert udiv_rne_tte_u32(14, 4) == 4  # 3.5 -> 4 (tie to even)
    print("udiv_rne_tte_u32 tests passed")

    print("All primitive tests passed!")
