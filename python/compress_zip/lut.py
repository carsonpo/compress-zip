"""
Look-up tables for exp2 and RoPE.

Matches CUDA LUT generation in helpers.cu and attention.cu.
"""

import numpy as np
import math
from typing import List

# Constants
EXP_FRAC_BITS = 8
EXP_FRAC_SIZE = 1 << EXP_FRAC_BITS  # 256
EXP_Q16_SCALE = 1 << 16  # Q16 format

ROPE_HEAD_DIM = 64
ROPE_HALF_DIM = ROPE_HEAD_DIM // 2
ROPE_DEFAULT_MAX_SEQ_LEN = 64  # Matches TOKENS_PER_CHUNK for chunked processing
ROPE_MAX_SEQ_LEN = ROPE_DEFAULT_MAX_SEQ_LEN  # Alias for backwards compatibility
ROPE_Q15_SCALE = 1 << 15  # Q1.15 format
ROPE_BASE = 10000.0


class Exp2LutQ16:
    """
    256-entry exp2 fractional LUT in Q16 format.

    exp2_lut[i] = round(2^(-i/256) * 65536)  # NEGATIVE exponent matching CUDA

    Used for computing exp2(-x) where x has 8 fractional bits.
    This matches CUDA softmax_cumfreq.cu exactly.
    """

    def __init__(self):
        self.table: List[int] = []
        self._generate()

    def _generate(self):
        """Generate the exp2 LUT matching CUDA negative exponent formula."""
        self.table = []
        for i in range(EXP_FRAC_SIZE):
            # CUDA uses NEGATIVE exponent: 2^(-i/256) in Q16
            # softmax_cumfreq.cu: double v = std::pow(2.0, -frac) * 65536.0;
            frac = i / EXP_FRAC_SIZE
            val = math.pow(2.0, -frac) * EXP_Q16_SCALE
            # Use ties-to-even rounding to match CUDA llrint
            q = round(val)  # Python 3 round uses ties-to-even
            q = max(1, min(65535, int(q)))
            self.table.append(q)

    def __getitem__(self, idx: int) -> int:
        return self.table[idx & 0xFF]

    def as_array(self) -> np.ndarray:
        return np.array(self.table, dtype=np.int32)


class RopeLut:
    """
    RoPE (Rotary Position Embedding) cos/sin LUT.

    Shape: [MAX_SEQ_LEN, HALF_DIM] for both cos and sin.
    Values in Q1.15 format (multiply by 32768).

    theta_i = pos / (10000 ^ (2i / head_dim))
    cos_lut[pos][i] = round(cos(theta_i) * 32768)
    sin_lut[pos][i] = round(sin(theta_i) * 32768)
    """

    def __init__(self, max_seq_len: int = ROPE_DEFAULT_MAX_SEQ_LEN, head_dim: int = ROPE_HEAD_DIM, base: float = ROPE_BASE):
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        self.base = base
        self.cos_table: np.ndarray = np.zeros((max_seq_len, self.half_dim), dtype=np.int16)
        self.sin_table: np.ndarray = np.zeros((max_seq_len, self.half_dim), dtype=np.int16)
        self._generate()

    def _generate(self):
        """Generate the RoPE LUTs."""
        for pos in range(self.max_seq_len):
            for i in range(self.half_dim):
                # Compute frequency: 1 / (base ^ (2i / head_dim))
                freq = 1.0 / math.pow(self.base, (2 * i) / self.head_dim)
                theta = pos * freq

                # Store in Q1.15
                cos_val = int(round(math.cos(theta) * ROPE_Q15_SCALE))
                sin_val = int(round(math.sin(theta) * ROPE_Q15_SCALE))

                # Clamp to i16 range
                cos_val = max(-32768, min(32767, cos_val))
                sin_val = max(-32768, min(32767, sin_val))

                self.cos_table[pos, i] = cos_val
                self.sin_table[pos, i] = sin_val

    def get_cos(self, pos: int, idx: int) -> int:
        return int(self.cos_table[pos, idx])

    def get_sin(self, pos: int, idx: int) -> int:
        return int(self.sin_table[pos, idx])


def exp_q16_from_neg_fixed(neg_x_q8: int, exp_lut: Exp2LutQ16) -> int:
    """
    Compute exp2(-x) where x is in Q8 fixed point (8 fractional bits).

    This computes 2^(-x) using the LUT for the fractional part.

    Args:
        neg_x_q8: The negative exponent in Q8 format (x >= 0, we compute 2^(-x))
        exp_lut: The exp2 LUT

    Returns:
        Result in Q16 format, or 0 if underflow
    """
    if neg_x_q8 < 0:
        # x is positive, so -x is negative, meaning 2^(-x) > 1
        # This shouldn't happen in softmax context
        neg_x_q8 = 0

    # Integer and fractional parts
    int_part = neg_x_q8 >> EXP_FRAC_BITS  # How many times to divide by 2
    frac_part = neg_x_q8 & (EXP_FRAC_SIZE - 1)

    # If int_part >= 16, result underflows to 0 in Q16
    if int_part >= 16:
        return 0

    # Get fractional exp2 from LUT
    frac_exp = exp_lut[frac_part]

    # Divide by 2^int_part (shift right)
    result = frac_exp >> int_part

    return result


# Pre-generated singleton instances
_exp2_lut: Exp2LutQ16 | None = None
_rope_lut: RopeLut | None = None


def get_exp2_lut() -> Exp2LutQ16:
    """Get the singleton exp2 LUT instance."""
    global _exp2_lut
    if _exp2_lut is None:
        _exp2_lut = Exp2LutQ16()
    return _exp2_lut


def get_rope_lut() -> RopeLut:
    """Get the singleton RoPE LUT instance."""
    global _rope_lut
    if _rope_lut is None:
        _rope_lut = RopeLut()
    return _rope_lut


# Test cases
if __name__ == "__main__":
    # Test Exp2LutQ16 - now uses NEGATIVE exponent: 2^(-i/256)
    lut = Exp2LutQ16()
    # At i=0, 2^(-0) = 1.0 -> 65536 in Q16 (clamped to 65535)
    assert lut[0] == 65535, f"Expected 65535, got {lut[0]}"
    # At i=128, 2^(-0.5) = 1/sqrt(2) ≈ 0.7071
    expected_half = int(round(math.pow(2, -0.5) * 65536))
    assert abs(lut[128] - expected_half) <= 1, f"Expected ~{expected_half}, got {lut[128]}"
    # At i=255, 2^(-255/256) ≈ 0.5027
    expected_near_one = int(round(math.pow(2, -255/256) * 65536))
    assert abs(lut[255] - expected_near_one) <= 1
    # Table should be monotonically decreasing (exp(-x) decreases as x increases)
    for i in range(1, EXP_FRAC_SIZE):
        assert lut[i] <= lut[i-1], f"Table not decreasing at {i}"
    print("Exp2LutQ16 tests passed")

    # Test RopeLut
    rope = RopeLut()
    # At position 0, all angles are 0, so cos=1, sin=0
    # Note: 1.0 * 32768 = 32768, which is clamped to 32767 in Q1.15
    assert rope.get_cos(0, 0) == 32767, f"Expected 32767, got {rope.get_cos(0, 0)}"
    assert rope.get_sin(0, 0) == 0  # sin(0) = 0
    print("RopeLut tests passed")

    # Test exp_q16_from_neg_fixed with the negative exponent LUT
    # Note: This function now works with the negative exponent LUT
    result = exp_q16_from_neg_fixed(0, lut)  # 2^0 = 1 (via lut[0] / 2^0)
    assert result == 65535, f"Expected 65535, got {result}"
    result = exp_q16_from_neg_fixed(256, lut)  # int_part=1, frac_part=0 -> lut[0] >> 1
    assert result == 32767, f"Expected 32767, got {result}"
    result = exp_q16_from_neg_fixed(512, lut)  # int_part=2, frac_part=0 -> lut[0] >> 2
    assert result == 16383, f"Expected 16383, got {result}"
    print("exp_q16_from_neg_fixed tests passed")

    print("All LUT tests passed!")
