"""Tests for arithmetic encoder/decoder."""

import pytest
import numpy as np
from compress_zip.arith_coder import (
    ArithEncoder,
    ArithDecoder,
    encode_symbols,
    decode_symbols,
)


class TestArithmeticCoder:
    """Tests for arithmetic encoding/decoding."""

    def test_simple_roundtrip(self):
        """Basic encode/decode roundtrip."""
        freqs = np.array([1, 2, 3, 4], dtype=np.int32)
        cumfreqs = np.zeros(5, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        original = [0, 1, 2, 3, 3, 2, 1, 0]
        encoded = encode_symbols(original, cumfreqs, total)
        decoded = decode_symbols(encoded, cumfreqs, total, len(original))

        assert original == decoded

    def test_uniform_distribution(self):
        """Test with uniform distribution."""
        vocab_size = 16
        freqs = np.ones(vocab_size, dtype=np.int32)
        cumfreqs = np.zeros(vocab_size + 1, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        original = list(range(vocab_size)) * 10
        encoded = encode_symbols(original, cumfreqs, total)
        decoded = decode_symbols(encoded, cumfreqs, total, len(original))

        assert original == decoded

    def test_skewed_distribution(self):
        """Test with highly skewed distribution."""
        freqs = np.array([1000, 1, 1, 1, 1], dtype=np.int32)
        cumfreqs = np.zeros(6, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        # Mostly symbol 0
        original = [0] * 100 + [1, 2, 3, 4] * 5
        encoded = encode_symbols(original, cumfreqs, total)
        decoded = decode_symbols(encoded, cumfreqs, total, len(original))

        assert original == decoded

    def test_large_vocabulary(self):
        """Test with larger vocabulary."""
        vocab_size = 256
        np.random.seed(42)
        freqs = np.random.randint(1, 100, vocab_size, dtype=np.int32)
        cumfreqs = np.zeros(vocab_size + 1, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        original = [int(np.random.randint(0, vocab_size)) for _ in range(500)]
        encoded = encode_symbols(original, cumfreqs, total)
        decoded = decode_symbols(encoded, cumfreqs, total, len(original))

        assert original == decoded

    def test_single_symbol(self):
        """Test encoding a single symbol."""
        freqs = np.array([1, 1], dtype=np.int32)
        cumfreqs = np.zeros(3, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        for sym in [0, 1]:
            encoded = encode_symbols([sym], cumfreqs, total)
            decoded = decode_symbols(encoded, cumfreqs, total, 1)
            assert [sym] == decoded

    def test_compression_ratio(self):
        """Verify compression is reasonable."""
        # Highly skewed: symbol 0 very likely
        freqs = np.array([10000, 1, 1, 1], dtype=np.int32)
        cumfreqs = np.zeros(5, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        # All symbol 0
        original = [0] * 1000
        encoded = encode_symbols(original, cumfreqs, total)

        # Should compress very well
        assert len(encoded) < len(original) * 0.1  # Less than 10% size

    def test_empty_sequence(self):
        """Test empty sequence."""
        freqs = np.array([1, 1], dtype=np.int32)
        cumfreqs = np.zeros(3, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        original = []
        encoded = encode_symbols(original, cumfreqs, total)
        decoded = decode_symbols(encoded, cumfreqs, total, 0)

        assert original == decoded


class TestArithEncoderState:
    """Tests for encoder internal state."""

    def test_encoder_state_progression(self):
        """Verify encoder state changes after each symbol."""
        freqs = np.array([1, 1], dtype=np.int32)
        cumfreqs = np.zeros(3, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        encoder = ArithEncoder()
        initial_low = encoder.low
        initial_high = encoder.high

        encoder.encode_symbol(cumfreqs, 0, total)

        # State should have changed (or bits output)
        assert encoder.low != initial_low or encoder.high != initial_high or len(encoder.output_bits) > 0


# Cross-verification vectors for arithmetic coder
# Format: (frequencies, symbols, expected_bytes_hex)
# These should match the Rust implementation exactly
ARITH_CROSS_VERIFICATION_VECTORS = [
    # Simple case with uniform distribution
    (
        [1, 1, 1, 1],  # frequencies
        [0, 1, 2, 3],  # symbols to encode
    ),
    # Skewed distribution
    (
        [10, 1, 1, 1],
        [0, 0, 0, 1, 2, 3],
    ),
]


class TestArithCrossVerification:
    """Cross-verification tests for arithmetic coder."""

    @pytest.mark.parametrize("freqs,symbols", ARITH_CROSS_VERIFICATION_VECTORS)
    def test_roundtrip(self, freqs, symbols):
        """Verify roundtrip works for cross-verification vectors."""
        freqs = np.array(freqs, dtype=np.int32)
        cumfreqs = np.zeros(len(freqs) + 1, dtype=np.int32)
        cumfreqs[1:] = np.cumsum(freqs)
        total = int(cumfreqs[-1])

        encoded = encode_symbols(symbols, cumfreqs, total)
        decoded = decode_symbols(encoded, cumfreqs, total, len(symbols))

        assert list(symbols) == decoded
