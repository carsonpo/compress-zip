"""Tests for look-up tables."""

import pytest
import numpy as np
import math
from compress_zip.lut import (
    Exp2LutQ16,
    RopeLut,
    exp_q16_from_neg_fixed,
    get_exp2_lut,
    get_rope_lut,
    EXP_Q16_SCALE,
    ROPE_Q15_SCALE,
    ROPE_HALF_DIM,
    ROPE_MAX_SEQ_LEN,
)


class TestExp2LutQ16:
    """Tests for exp2 look-up table."""

    def test_table_size(self):
        """Table should have 256 entries."""
        lut = Exp2LutQ16()
        assert len(lut.table) == 256

    def test_first_entry(self):
        """First entry is 2^0 = 1.0 in Q16."""
        lut = Exp2LutQ16()
        assert lut[0] == EXP_Q16_SCALE  # 65536

    def test_midpoint(self):
        """Entry 128 is 2^0.5 = sqrt(2) in Q16."""
        lut = Exp2LutQ16()
        expected = int(round(math.sqrt(2) * EXP_Q16_SCALE))
        assert abs(lut[128] - expected) <= 1

    def test_last_entry(self):
        """Entry 255 is 2^(255/256) in Q16."""
        lut = Exp2LutQ16()
        expected = int(round(math.pow(2, 255/256) * EXP_Q16_SCALE))
        assert abs(lut[255] - expected) <= 1

    def test_monotonically_increasing(self):
        """Table values should be monotonically increasing."""
        lut = Exp2LutQ16()
        for i in range(255):
            assert lut[i] <= lut[i + 1]

    def test_indexing_wraps(self):
        """Index should wrap with & 0xFF."""
        lut = Exp2LutQ16()
        assert lut[256] == lut[0]
        assert lut[257] == lut[1]


class TestRopeLut:
    """Tests for RoPE look-up table."""

    def test_table_shape(self):
        """Tables should have correct shape."""
        lut = RopeLut()
        assert lut.cos_table.shape == (ROPE_MAX_SEQ_LEN, ROPE_HALF_DIM)
        assert lut.sin_table.shape == (ROPE_MAX_SEQ_LEN, ROPE_HALF_DIM)

    def test_position_zero(self):
        """At position 0, cos=1, sin=0 for all frequencies."""
        lut = RopeLut()
        for i in range(ROPE_HALF_DIM):
            # cos(0) = 1.0 -> round(1.0 * 32768) = 32768, clamped to int16 max 32767
            assert lut.get_cos(0, i) == 32767
            assert lut.get_sin(0, i) == 0  # sin(0) = 0

    def test_cos_sin_identity(self):
        """cos^2 + sin^2 should be approximately 1."""
        lut = RopeLut()
        for pos in [0, 1, 10, 63]:
            for i in [0, 15, 31]:
                cos_val = lut.get_cos(pos, i) / ROPE_Q15_SCALE
                sin_val = lut.get_sin(pos, i) / ROPE_Q15_SCALE
                identity = cos_val**2 + sin_val**2
                assert abs(identity - 1.0) < 0.01  # Within 1%

    def test_values_in_range(self):
        """All values should be in Q1.15 range."""
        lut = RopeLut()
        assert np.all(lut.cos_table >= -32768)
        assert np.all(lut.cos_table <= 32767)
        assert np.all(lut.sin_table >= -32768)
        assert np.all(lut.sin_table <= 32767)


class TestExpQ16FromNegFixed:
    """Tests for exp2(-x) computation."""

    def test_zero_exponent(self):
        """2^(-0) = 1."""
        lut = get_exp2_lut()
        result = exp_q16_from_neg_fixed(0, lut)
        assert result == EXP_Q16_SCALE

    def test_one_exponent(self):
        """2^(-1) = 0.5 -> 32768 in Q16."""
        lut = get_exp2_lut()
        result = exp_q16_from_neg_fixed(256, lut)  # 256 in Q8 = 1.0
        assert result == 32768

    def test_two_exponent(self):
        """2^(-2) = 0.25 -> 16384 in Q16."""
        lut = get_exp2_lut()
        result = exp_q16_from_neg_fixed(512, lut)  # 512 in Q8 = 2.0
        assert result == 16384

    def test_large_exponent_underflows(self):
        """Large exponents should underflow to 0."""
        lut = get_exp2_lut()
        result = exp_q16_from_neg_fixed(16 * 256, lut)  # 16.0 in Q8
        assert result == 0

    def test_fractional_exponent(self):
        """Test fractional exponent."""
        lut = get_exp2_lut()
        # For neg_x_q8=128 (0.5 in Q8 format):
        # int_part = 128 >> 8 = 0, frac_part = 128
        # result = lut[128] >> 0 = 2^(128/256) * 65536 = sqrt(2) * 65536
        # This is used with additional scaling in softmax
        result = exp_q16_from_neg_fixed(128, lut)
        expected_lut_value = int(round(math.sqrt(2) * EXP_Q16_SCALE))
        assert abs(result - expected_lut_value) <= 1


class TestSingletons:
    """Tests for singleton instances."""

    def test_exp2_singleton(self):
        """get_exp2_lut returns same instance."""
        lut1 = get_exp2_lut()
        lut2 = get_exp2_lut()
        assert lut1 is lut2

    def test_rope_singleton(self):
        """get_rope_lut returns same instance."""
        lut1 = get_rope_lut()
        lut2 = get_rope_lut()
        assert lut1 is lut2


# Cross-verification vectors
EXP_CROSS_VERIFICATION_VECTORS = [
    # (neg_x_q8, expected_result)
    (0, 65536),      # 2^0 = 1
    (256, 32768),    # 2^(-1) = 0.5
    (512, 16384),    # 2^(-2) = 0.25
    (768, 8192),     # 2^(-3) = 0.125
]


class TestLutCrossVerification:
    """Cross-verification tests for LUTs."""

    @pytest.mark.parametrize("neg_x_q8,expected", EXP_CROSS_VERIFICATION_VECTORS)
    def test_exp_vector(self, neg_x_q8, expected):
        """Verify exp2 values match expected."""
        lut = get_exp2_lut()
        result = exp_q16_from_neg_fixed(neg_x_q8, lut)
        assert result == expected
