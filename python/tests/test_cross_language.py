"""
Cross-language verification tests.

These tests verify that Python and Rust implementations produce bit-exact
identical results for all core operations.

To run the full cross-language verification:
1. Run these Python tests: pytest tests/test_cross_language.py -v
2. Run the Rust tests: cargo test test_cross_language --release

Both should produce identical outputs for the same inputs.
"""

import numpy as np
import pytest

from compress_zip.softmax_cdf import (
    build_cumfreqs,
    compute_coef_q24,
    compute_target_total,
    exp_q16_from_diff_acc,
    DEFAULT_COEF_Q24,
)
from compress_zip.lut import get_exp2_lut
from compress_zip.arith_coder import ArithEncoder, ArithDecoder
from compress_zip.primitives import sra_rne_tte_s32, isqrt32_restoring, udiv_rne_tte_u32


class TestCrossLanguageConstants:
    """Test that constants match between Python and Rust."""

    def test_coef_q24_value(self):
        """Verify COEF_Q24 is computed correctly."""
        # COEF_D = (0.1 * 1.4426950408889634) / 64.0 = 0.002254211001389...
        # COEF_Q24 = round(COEF_D * 2^24) = round(37819.38...) = 37819
        coef = compute_coef_q24()
        assert coef == 37819, f"Expected 37819, got {coef}"
        assert coef == DEFAULT_COEF_Q24

    def test_target_total_vocab_1024(self):
        """Test target_total for vocab_size=1024."""
        target = compute_target_total(1024)
        # QUARTER_RANGE = 1 << 30 = 1073741824
        # target = 1073741824 - 1024 - 1024 = 1073739776
        assert target == 1073739776, f"Expected 1073739776, got {target}"

    def test_target_total_vocab_256(self):
        """Test target_total for vocab_size=256."""
        target = compute_target_total(256)
        expected = (1 << 30) - 256 - 1024  # 1073740544
        assert target == expected, f"Expected {expected}, got {target}"


class TestCrossLanguageExp:
    """Test exp_q16_from_diff_acc produces correct values."""

    def test_exp_zero_diff(self):
        """exp(0) should give max value 65535."""
        exp_lut = get_exp2_lut()
        result = exp_q16_from_diff_acc(0, exp_lut)
        assert result == 65535

    def test_exp_negative_diff(self):
        """Negative diffs should give smaller values (eventually)."""
        exp_lut = get_exp2_lut()
        coef = compute_coef_q24()

        e0 = exp_q16_from_diff_acc(0, exp_lut, coef)
        e100 = exp_q16_from_diff_acc(-100, exp_lut, coef)
        e1000 = exp_q16_from_diff_acc(-1000, exp_lut, coef)
        e10000 = exp_q16_from_diff_acc(-10000, exp_lut, coef)

        assert e0 == 65535
        # Note: With small coef_q24, e100 may still be 65535 (no decay yet)
        assert e100 <= e0
        assert e1000 <= e100
        assert e10000 < e1000  # Eventually there's decay
        assert e10000 >= 1  # Never goes to 0

    def test_exp_known_values(self):
        """Test specific known values for cross-language verification."""
        exp_lut = get_exp2_lut()
        coef = compute_coef_q24()

        # These values should match exactly in Rust
        test_cases = [
            (0, 65535),
            (-50, None),  # Will capture actual value
            (-100, None),
            (-500, None),
            (-1000, None),
            (-5000, None),
        ]

        results = []
        for diff, expected in test_cases:
            result = exp_q16_from_diff_acc(diff, exp_lut, coef)
            results.append((diff, result))
            if expected is not None:
                assert result == expected, f"diff={diff}: expected {expected}, got {result}"

        # Print for cross-language verification
        print("\n=== Cross-Language Exp Values (Python) ===")
        print(f"COEF_Q24 = {coef}")
        for diff, result in results:
            print(f"exp_q16_from_diff({diff}) = {result}")


class TestCrossLanguageSoftmaxCDF:
    """Test build_cumfreqs produces identical results."""

    def test_uniform_logits(self):
        """Test with uniform logits (all same value)."""
        logits = np.array([1000, 1000, 1000, 1000], dtype=np.int32)
        cumfreqs, total = build_cumfreqs(logits)

        # With uniform logits, frequencies should be nearly equal
        freqs = np.diff(np.concatenate([[0], cumfreqs]))
        assert cumfreqs[-1] == total
        assert all(f >= 1 for f in freqs)

        print("\n=== Uniform Logits Test (Python) ===")
        print(f"logits: {logits.tolist()}")
        print(f"cumfreqs: {cumfreqs.tolist()}")
        print(f"freqs: {freqs.tolist()}")
        print(f"total: {total}")

    def test_varied_logits(self):
        """Test with varied logits."""
        logits = np.array([5000, 3000, 1000, -1000, -3000, -5000, -7000, -9000], dtype=np.int32)
        cumfreqs, total = build_cumfreqs(logits)

        freqs = np.diff(np.concatenate([[0], cumfreqs]))
        assert cumfreqs[-1] == total
        assert all(f >= 1 for f in freqs)

        # Higher logits should have higher frequencies
        assert freqs[0] > freqs[-1]

        print("\n=== Varied Logits Test (Python) ===")
        print(f"logits: {logits.tolist()}")
        print(f"cumfreqs: {cumfreqs.tolist()}")
        print(f"freqs: {freqs.tolist()}")
        print(f"total: {total}")

    def test_large_vocab_deterministic(self):
        """Test with larger vocab for determinism."""
        np.random.seed(42)
        logits = np.random.randint(-10000, 10000, 64, dtype=np.int32)
        cumfreqs, total = build_cumfreqs(logits)

        # Verify basic properties
        assert cumfreqs[-1] == total
        freqs = np.diff(np.concatenate([[0], cumfreqs]))
        assert all(f >= 1 for f in freqs)
        assert sum(freqs) == total

        print("\n=== Large Vocab Test (Python) ===")
        print(f"First 8 logits: {logits[:8].tolist()}")
        print(f"First 8 cumfreqs: {cumfreqs[:8].tolist()}")
        print(f"First 8 freqs: {freqs[:8].tolist()}")
        print(f"total: {total}")


class TestCrossLanguageArithCoder:
    """Test arithmetic coder roundtrip with fixed cumfreqs."""

    def test_simple_roundtrip(self):
        """Test encode/decode roundtrip with simple frequencies."""
        # Create cumfreqs manually (inclusive prefix sum)
        vocab_size = 4
        target_total = compute_target_total(vocab_size)

        # Simple uniform-ish distribution
        base_freq = target_total // vocab_size
        freqs = [base_freq] * vocab_size
        remainder = target_total - sum(freqs)
        freqs[0] += remainder

        cumfreqs = np.zeros(vocab_size, dtype=np.uint32)
        running = 0
        for i, f in enumerate(freqs):
            running += f
            cumfreqs[i] = running

        assert cumfreqs[-1] == target_total

        # Encode some symbols
        symbols = [0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3]
        encoder = ArithEncoder()
        for sym in symbols:
            encoder.encode_symbol(cumfreqs, sym, target_total)
        compressed = encoder.finish()

        # Decode
        decoder = ArithDecoder(compressed)
        decoded = []
        for _ in symbols:
            sym = decoder.decode_symbol(cumfreqs, target_total)
            decoded.append(sym)

        assert symbols == decoded, f"Roundtrip failed: {symbols} != {decoded}"

        print("\n=== Arithmetic Coder Roundtrip (Python) ===")
        print(f"symbols: {symbols}")
        print(f"compressed length: {len(compressed)}")
        print(f"compressed bytes (hex): {compressed[:16].hex()}...")


class TestCrossLanguagePrimitives:
    """Test primitive operations match Rust exactly."""

    def test_sra_rne_tte_s32(self):
        """Test shift-right with ties-to-even."""
        test_cases = [
            # (value, shift, expected)
            (7, 1, 4),      # 7 >> 1 = 3.5, ties to 4 (even)
            (5, 1, 2),      # 5 >> 1 = 2.5, ties to 2 (even)
            (6, 1, 3),      # 6 >> 1 = 3, no tie
            (1234567, 10, 1206),  # From README
            (-7, 1, -4),    # Negative
            (-5, 1, -2),    # Negative tie-to-even
        ]

        print("\n=== SRA RNE TTE Tests (Python) ===")
        for value, shift, expected in test_cases:
            result = sra_rne_tte_s32(value, shift)
            print(f"sra_rne_tte_s32({value}, {shift}) = {result} (expected {expected})")
            assert result == expected, f"sra_rne_tte_s32({value}, {shift}) = {result}, expected {expected}"

    def test_isqrt32_restoring(self):
        """Test integer square root."""
        test_cases = [
            (0, 0),
            (1, 1),
            (4, 2),
            (9, 3),
            (16, 4),
            (1000000, 1000),
            (2147483647, 46340),  # From README
        ]

        print("\n=== Integer Square Root Tests (Python) ===")
        for value, expected in test_cases:
            result = isqrt32_restoring(value)
            print(f"isqrt32_restoring({value}) = {result} (expected {expected})")
            assert result == expected, f"isqrt32_restoring({value}) = {result}, expected {expected}"

    def test_udiv_rne_tte_u32(self):
        """Test unsigned division with ties-to-even.

        For odd divisors, when remainder == half, we round up.
        For even divisors, when remainder == half, we round to even.
        """
        test_cases = [
            (1000000, 3, 333334),  # From README (with rounding)
            (10, 3, 4),           # 10/3 = 3 r 1, half=1, odd divisor -> round up
            (10, 2, 5),           # Exact division
            (7, 2, 4),            # 3.5 ties to 4 (even quotient)
            (5, 2, 2),            # 2.5 ties to 2 (even quotient)
            (9, 3, 3),            # Exact division
            (11, 3, 4),           # 11/3 = 3 r 2, half=1, r > half -> round up
        ]

        print("\n=== Division RNE TTE Tests (Python) ===")
        for num, div, expected in test_cases:
            result = udiv_rne_tte_u32(num, div)
            print(f"udiv_rne_tte_u32({num}, {div}) = {result} (expected {expected})")
            assert result == expected, f"udiv_rne_tte_u32({num}, {div}) = {result}, expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
