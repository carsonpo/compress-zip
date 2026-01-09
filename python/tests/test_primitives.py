"""Tests for deterministic integer primitives."""

import pytest
import numpy as np
from compress_zip.primitives import (
    sra_rne_tte_s32,
    sra_rne_tte_s64_to_s32,
    udiv_rne_tte_u32,
    div_rne_tte_s32,
    isqrt32_restoring,
    clamp_i8,
    argmax_deterministic,
    sra_rne_tte_s32_np,
    clamp_i8_np,
)


class TestSraRneTteS32:
    """Tests for shift right arithmetic with ties-to-even."""

    def test_no_shift(self):
        """Shift by 0 returns original value."""
        assert sra_rne_tte_s32(100, 0) == 100
        assert sra_rne_tte_s32(-100, 0) == -100

    def test_shift_32_or_more(self):
        """Large shifts return 0."""
        assert sra_rne_tte_s32(12345, 32) == 0
        assert sra_rne_tte_s32(12345, 64) == 0

    def test_exact_division(self):
        """Exact division (no rounding needed)."""
        assert sra_rne_tte_s32(8, 1) == 4  # 8/2 = 4 exactly
        assert sra_rne_tte_s32(16, 2) == 4  # 16/4 = 4 exactly
        assert sra_rne_tte_s32(-8, 1) == -4

    def test_round_down(self):
        """Values below midpoint round down."""
        assert sra_rne_tte_s32(9, 2) == 2  # 9/4 = 2.25 -> 2

    def test_round_up(self):
        """Values above midpoint round up."""
        assert sra_rne_tte_s32(11, 2) == 3  # 11/4 = 2.75 -> 3

    def test_ties_to_even(self):
        """Ties round to even result."""
        # 7 >> 1 = 3.5, rounds to 4 (even)
        assert sra_rne_tte_s32(7, 1) == 4
        # 5 >> 1 = 2.5, rounds to 2 (even)
        assert sra_rne_tte_s32(5, 1) == 2
        # 3 >> 1 = 1.5, rounds to 2 (even)
        assert sra_rne_tte_s32(3, 1) == 2
        # 1 >> 1 = 0.5, rounds to 0 (even)
        assert sra_rne_tte_s32(1, 1) == 0

    def test_negative_ties_to_even(self):
        """Negative ties also round to even."""
        assert sra_rne_tte_s32(-7, 1) == -4
        assert sra_rne_tte_s32(-5, 1) == -2


class TestIsqrt32Restoring:
    """Tests for integer square root."""

    def test_zero(self):
        assert isqrt32_restoring(0) == 0

    def test_perfect_squares(self):
        assert isqrt32_restoring(1) == 1
        assert isqrt32_restoring(4) == 2
        assert isqrt32_restoring(9) == 3
        assert isqrt32_restoring(16) == 4
        assert isqrt32_restoring(25) == 5
        assert isqrt32_restoring(100) == 10
        assert isqrt32_restoring(10000) == 100
        assert isqrt32_restoring(1000000) == 1000

    def test_non_perfect_squares(self):
        """Non-perfect squares return floor of sqrt."""
        assert isqrt32_restoring(2) == 1
        assert isqrt32_restoring(3) == 1
        assert isqrt32_restoring(5) == 2
        assert isqrt32_restoring(8) == 2
        assert isqrt32_restoring(10) == 3
        assert isqrt32_restoring(15) == 3
        assert isqrt32_restoring(17) == 4

    def test_large_values(self):
        """Test with large values."""
        assert isqrt32_restoring(2**30) == 2**15
        # Max u32 value
        assert isqrt32_restoring(2**32 - 1) == 65535


class TestUdivRneTteU32:
    """Tests for unsigned division with ties-to-even."""

    def test_exact_division(self):
        assert udiv_rne_tte_u32(10, 2) == 5
        assert udiv_rne_tte_u32(100, 10) == 10

    def test_round_down(self):
        assert udiv_rne_tte_u32(10, 4) == 2  # 2.5 -> 2 (tie to even)

    def test_round_up(self):
        assert udiv_rne_tte_u32(11, 4) == 3  # 2.75 -> 3

    def test_ties_to_even(self):
        # 7/2 = 3.5 -> 4 (even)
        assert udiv_rne_tte_u32(7, 2) == 4
        # 5/2 = 2.5 -> 2 (even)
        assert udiv_rne_tte_u32(5, 2) == 2
        # 14/4 = 3.5 -> 4 (even)
        assert udiv_rne_tte_u32(14, 4) == 4
        # 10/4 = 2.5 -> 2 (even)
        assert udiv_rne_tte_u32(10, 4) == 2

    def test_division_by_zero_raises(self):
        with pytest.raises(ValueError):
            udiv_rne_tte_u32(10, 0)


class TestClampI8:
    """Tests for int8 clamping."""

    def test_in_range(self):
        assert clamp_i8(0) == 0
        assert clamp_i8(100) == 100
        assert clamp_i8(-100) == -100
        assert clamp_i8(127) == 127
        assert clamp_i8(-128) == -128

    def test_overflow(self):
        assert clamp_i8(128) == 127
        assert clamp_i8(1000) == 127

    def test_underflow(self):
        assert clamp_i8(-129) == -128
        assert clamp_i8(-1000) == -128


class TestArgmaxDeterministic:
    """Tests for deterministic argmax."""

    def test_single_max(self):
        assert argmax_deterministic([1, 5, 3, 2]) == 1
        assert argmax_deterministic([5, 1, 3, 2]) == 0
        assert argmax_deterministic([1, 3, 2, 5]) == 3

    def test_ties_return_lowest_index(self):
        """On ties, return lowest index."""
        assert argmax_deterministic([5, 5, 3, 2]) == 0
        assert argmax_deterministic([1, 5, 5, 2]) == 1
        assert argmax_deterministic([5, 5, 5, 5]) == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            argmax_deterministic([])


class TestNumpyVectorized:
    """Tests for numpy vectorized versions."""

    def test_sra_rne_tte_s32_np(self):
        x = np.array([7, 5, 3, 1], dtype=np.int32)
        result = sra_rne_tte_s32_np(x, 1)
        expected = np.array([4, 2, 2, 0], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_clamp_i8_np(self):
        x = np.array([-200, -128, 0, 127, 200], dtype=np.int32)
        result = clamp_i8_np(x)
        expected = np.array([-128, -128, 0, 127, 127], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)


# Cross-verification test vectors
# These should match the Rust implementation exactly
CROSS_VERIFICATION_VECTORS = [
    # (function, args, expected)
    ("sra_rne_tte_s32", (7, 1), 4),
    ("sra_rne_tte_s32", (5, 1), 2),
    ("sra_rne_tte_s32", (3, 1), 2),
    ("sra_rne_tte_s32", (1, 1), 0),
    ("sra_rne_tte_s32", (-7, 1), -4),
    ("sra_rne_tte_s32", (1234567, 10), 1206),  # 1205.63 rounds up
    ("isqrt32_restoring", (1000000,), 1000),
    ("isqrt32_restoring", (2147483647,), 46340),
    ("udiv_rne_tte_u32", (7, 2), 4),
    ("udiv_rne_tte_u32", (5, 2), 2),
    ("udiv_rne_tte_u32", (1000000, 3), 333334),  # 333333.33 rounds up
]


class TestCrossVerification:
    """Test vectors for cross-language verification."""

    @pytest.mark.parametrize("func_name,args,expected", CROSS_VERIFICATION_VECTORS)
    def test_vector(self, func_name, args, expected):
        func_map = {
            "sra_rne_tte_s32": sra_rne_tte_s32,
            "isqrt32_restoring": isqrt32_restoring,
            "udiv_rne_tte_u32": udiv_rne_tte_u32,
        }
        func = func_map[func_name]
        result = func(*args)
        assert result == expected, f"{func_name}{args} = {result}, expected {expected}"
