"""
Arithmetic encoder/decoder for neural compression.

Matches the CUDA/C++ arithmetic coder with 32-bit state.
Uses E1/E2/E3 renormalization scheme.
"""

import numpy as np
from typing import List, Tuple
from io import BytesIO


# Constants matching CUDA implementation
NUM_STATE_BITS = 32
HALF_RANGE = 1 << 31  # 2^31
QUARTER_RANGE = 1 << 30  # 2^30
THREE_QUARTER_RANGE = 3 * QUARTER_RANGE

# Masks
MAX_RANGE = (1 << NUM_STATE_BITS) - 1  # 0xFFFFFFFF


class ArithEncoder:
    """
    Arithmetic encoder with 32-bit state.

    Uses E1/E2/E3 renormalization:
    - E1: low >= HALF, both in upper half -> output 1, then pending 0s
    - E2: high < HALF, both in lower half -> output 0, then pending 1s
    - E3: low in [1/4, 1/2), high in [1/2, 3/4) -> increment pending, rescale

    Bits are packed MSB-first into bytes.
    """

    def __init__(self):
        self.low: int = 0
        self.high: int = MAX_RANGE
        self.pending_bits: int = 0
        self.output_bits: List[int] = []

    def _output_bit(self, bit: int):
        """Output a single bit."""
        self.output_bits.append(bit)

    def _output_bit_with_pending(self, bit: int):
        """Output a bit followed by pending opposite bits."""
        self._output_bit(bit)
        for _ in range(self.pending_bits):
            self._output_bit(1 - bit)
        self.pending_bits = 0

    def encode_symbol(self, cumfreqs: np.ndarray, symbol: int, total: int):
        """
        Encode a single symbol.

        Matches CUDA arith_cuda.cu inclusive cumsum format.

        Args:
            cumfreqs: Cumulative frequency table [vocab_size] (inclusive cumsum)
            symbol: Symbol index to encode (0 to vocab_size - 1)
            total: Total frequency (cumfreqs[-1])
        """
        range_size = self.high - self.low + 1

        # Update range based on symbol (CUDA-matching inclusive format)
        # sym_low = cumfreqs[symbol - 1] if symbol > 0 else 0
        # sym_high = cumfreqs[symbol]
        sym_high = int(cumfreqs[symbol])
        sym_low = int(cumfreqs[symbol - 1]) if symbol > 0 else 0

        # new_high = low + (range * sym_high / total) - 1
        # new_low = low + (range * sym_low / total)
        self.high = self.low + (range_size * sym_high // total) - 1
        self.low = self.low + (range_size * sym_low // total)

        # Renormalization loop
        while True:
            if self.high < HALF_RANGE:
                # E2: Both in lower half
                self._output_bit_with_pending(0)
                self.low = self.low << 1
                self.high = (self.high << 1) | 1
            elif self.low >= HALF_RANGE:
                # E1: Both in upper half
                self._output_bit_with_pending(1)
                self.low = (self.low - HALF_RANGE) << 1
                self.high = ((self.high - HALF_RANGE) << 1) | 1
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                # E3: Straddling the middle
                self.pending_bits += 1
                self.low = (self.low - QUARTER_RANGE) << 1
                self.high = ((self.high - QUARTER_RANGE) << 1) | 1
            else:
                break

            # Keep values in range
            self.low &= MAX_RANGE
            self.high &= MAX_RANGE

    def finish(self) -> bytes:
        """
        Finish encoding and return compressed bytes.

        Matches CUDA arith_encode_finish_kernel: emits a single '1' bit as terminator.
        """
        # Emit a fixed '1' bit as terminator (matching CUDA)
        self._output_bit(1)

        # Flush partial byte if any bits written
        if len(self.output_bits) % 8 != 0:
            # Pad remaining bits in current byte with zeros
            while len(self.output_bits) % 8 != 0:
                self.output_bits.append(0)

        # Convert bits to bytes (MSB first)
        result = bytearray()
        for i in range(0, len(self.output_bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.output_bits[i + j]
            result.append(byte)

        return bytes(result)


class ArithDecoder:
    """
    Arithmetic decoder with 32-bit state.

    Inverse of ArithEncoder.
    """

    def __init__(self, data: bytes):
        self.data = data
        self.bit_pos = 0
        self.low: int = 0
        self.high: int = MAX_RANGE
        self.value: int = 0

        # Initialize value from first 32 bits
        for _ in range(NUM_STATE_BITS):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        """Read the next bit from the input stream."""
        if self.bit_pos >= len(self.data) * 8:
            return 0  # Pad with zeros

        byte_idx = self.bit_pos // 8
        bit_idx = 7 - (self.bit_pos % 8)  # MSB first
        bit = (self.data[byte_idx] >> bit_idx) & 1
        self.bit_pos += 1
        return bit

    def decode_symbol(self, cumfreqs: np.ndarray, total: int) -> int:
        """
        Decode a single symbol.

        Matches CUDA arith_cuda.cu inclusive cumsum format.

        Args:
            cumfreqs: Cumulative frequency table [vocab_size] (inclusive cumsum)
            total: Total frequency (cumfreqs[-1])

        Returns:
            Decoded symbol index
        """
        range_size = self.high - self.low + 1

        # Find which symbol the current value falls into
        # scaled_value = ((value - low + 1) * total - 1) / range
        scaled = ((self.value - self.low + 1) * total - 1) // range_size

        # Binary search for symbol (CUDA-matching inclusive format)
        # For inclusive cumsum: symbol k spans [cumfreqs[k-1], cumfreqs[k])
        # where cumfreqs[-1] = 0 by convention
        vocab_size = len(cumfreqs)
        symbol = 0
        for i in range(vocab_size):
            if cumfreqs[i] > scaled:
                symbol = i
                break
        else:
            symbol = vocab_size - 1

        # Update range (CUDA-matching inclusive format)
        sym_high = int(cumfreqs[symbol])
        sym_low = int(cumfreqs[symbol - 1]) if symbol > 0 else 0

        self.high = self.low + (range_size * sym_high // total) - 1
        self.low = self.low + (range_size * sym_low // total)

        # Renormalization loop (same as encoder)
        while True:
            if self.high < HALF_RANGE:
                # E2: Both in lower half
                self.low = self.low << 1
                self.high = (self.high << 1) | 1
                self.value = (self.value << 1) | self._read_bit()
            elif self.low >= HALF_RANGE:
                # E1: Both in upper half
                self.low = (self.low - HALF_RANGE) << 1
                self.high = ((self.high - HALF_RANGE) << 1) | 1
                self.value = ((self.value - HALF_RANGE) << 1) | self._read_bit()
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                # E3: Straddling the middle
                self.low = (self.low - QUARTER_RANGE) << 1
                self.high = ((self.high - QUARTER_RANGE) << 1) | 1
                self.value = ((self.value - QUARTER_RANGE) << 1) | self._read_bit()
            else:
                break

            # Keep values in range
            self.low &= MAX_RANGE
            self.high &= MAX_RANGE
            self.value &= MAX_RANGE

        return symbol


def encode_symbols(symbols: List[int], cumfreqs: np.ndarray, total: int) -> bytes:
    """Convenience function to encode a list of symbols."""
    encoder = ArithEncoder()
    for sym in symbols:
        encoder.encode_symbol(cumfreqs, sym, total)
    return encoder.finish()


def decode_symbols(data: bytes, cumfreqs: np.ndarray, total: int, num_symbols: int) -> List[int]:
    """Convenience function to decode a list of symbols."""
    decoder = ArithDecoder(data)
    symbols = []
    for _ in range(num_symbols):
        symbols.append(decoder.decode_symbol(cumfreqs, total))
    return symbols


# Test cases
if __name__ == "__main__":
    # Test basic encode/decode roundtrip
    # Simple frequency table: 4 symbols with frequencies [1, 2, 3, 4]
    # Using CUDA-matching inclusive cumsum format (no leading 0)
    freqs = np.array([1, 2, 3, 4], dtype=np.uint32)
    cumfreqs = np.cumsum(freqs).astype(np.uint32)  # Inclusive: [1, 3, 6, 10]
    total = int(cumfreqs[-1])

    print(f"Frequencies: {freqs}")
    print(f"Cumfreqs (inclusive): {cumfreqs}")
    print(f"Total: {total}")

    # Encode some symbols
    original_symbols = [0, 1, 2, 3, 3, 2, 1, 0, 2, 2]
    encoded = encode_symbols(original_symbols, cumfreqs, total)
    print(f"Original: {original_symbols}")
    print(f"Encoded: {len(encoded)} bytes: {encoded.hex()}")

    # Decode
    decoded = decode_symbols(encoded, cumfreqs, total, len(original_symbols))
    print(f"Decoded: {decoded}")

    assert original_symbols == decoded, "Roundtrip failed!"
    print("Roundtrip test passed!")

    # Test with larger vocabulary
    vocab_size = 100
    np.random.seed(42)
    freqs_large = np.random.randint(1, 100, vocab_size, dtype=np.uint32)
    cumfreqs_large = np.cumsum(freqs_large).astype(np.uint32)  # Inclusive
    total_large = int(cumfreqs_large[-1])

    # Generate random symbols
    original_large = [int(np.random.randint(0, vocab_size)) for _ in range(1000)]

    encoded_large = encode_symbols(original_large, cumfreqs_large, total_large)
    decoded_large = decode_symbols(encoded_large, cumfreqs_large, total_large, len(original_large))

    assert original_large == decoded_large, "Large vocab roundtrip failed!"
    print(f"Large vocab test passed! {len(original_large)} symbols -> {len(encoded_large)} bytes")
    print(f"Compression ratio: {len(original_large) / len(encoded_large):.2f}x")

    print("\nAll arithmetic coder tests passed!")
