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
ROPE_MAX_SEQ_LEN = 64
ROPE_Q15_SCALE = 1 << 15  # Q1.15 format
ROPE_BASE = 10000.0


class Exp2LutQ16:
    """
    256-entry exp2 fractional LUT in Q16 format.

    exp2_lut[i] = round(2^(i/256) * 65536)

    Used for computing exp2(x) where x has 8 fractional bits.
    """

    def __init__(self):
        self.table: List[int] = []
        self._generate()

    def _generate(self):
        """Generate the exp2 LUT."""
        self.table = []
        for i in range(EXP_FRAC_SIZE):
            # 2^(i/256) in Q16
            val = math.pow(2.0, i / EXP_FRAC_SIZE) * EXP_Q16_SCALE
            self.table.append(int(round(val)))

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

    def __init__(self):
        self.cos_table: np.ndarray = np.zeros((ROPE_MAX_SEQ_LEN, ROPE_HALF_DIM), dtype=np.int16)
        self.sin_table: np.ndarray = np.zeros((ROPE_MAX_SEQ_LEN, ROPE_HALF_DIM), dtype=np.int16)
        self._generate()

    def _generate(self):
        """Generate the RoPE LUTs."""
        for pos in range(ROPE_MAX_SEQ_LEN):
            for i in range(ROPE_HALF_DIM):
                # Compute frequency: 1 / (base ^ (2i / head_dim))
                freq = 1.0 / math.pow(ROPE_BASE, (2 * i) / ROPE_HEAD_DIM)
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
    # Test Exp2LutQ16
    lut = Exp2LutQ16()
    assert lut[0] == 65536  # 2^0 = 1.0 in Q16
    assert abs(lut[128] - int(round(math.sqrt(2) * 65536))) <= 1  # 2^0.5
    assert abs(lut[255] - int(round(math.pow(2, 255/256) * 65536))) <= 1
    print("Exp2LutQ16 tests passed")

    # Test RopeLut
    rope = RopeLut()
    # At position 0, all angles are 0, so cos=1, sin=0
    assert rope.get_cos(0, 0) == 32768  # cos(0) = 1 in Q15
    assert rope.get_sin(0, 0) == 0  # sin(0) = 0
    print("RopeLut tests passed")

    # Test exp_q16_from_neg_fixed
    result = exp_q16_from_neg_fixed(0, lut)  # 2^0 = 1
    assert result == 65536
    result = exp_q16_from_neg_fixed(256, lut)  # 2^(-1) = 0.5
    assert result == 32768
    result = exp_q16_from_neg_fixed(512, lut)  # 2^(-2) = 0.25
    assert result == 16384
    print("exp_q16_from_neg_fixed tests passed")

    print("All LUT tests passed!")
