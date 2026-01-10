//! Arithmetic encoder and decoder with 32-bit state.
//!
//! Matches the CUDA implementation in arith_cuda.cu.

use crate::softmax_cdf::{HALF_RANGE, NUM_STATE_BITS, QUARTER_RANGE, STATE_MASK};

/// Arithmetic encoder state.
pub struct ArithEncoder {
    pub low: u64,
    pub high: u64,
    pub underflow: u32,
    /// Output bitstream (packed bytes, MSB first)
    pub output: Vec<u8>,
    /// Current byte being built
    pub current_byte: u8,
    /// Bit position in current byte (7 = MSB, 0 = LSB)
    pub bit_pos: i32,
}

impl Default for ArithEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithEncoder {
    /// Create a new encoder with initial state.
    pub fn new() -> Self {
        Self {
            low: 0,
            high: STATE_MASK,
            underflow: 0,
            output: Vec::new(),
            current_byte: 0,
            bit_pos: 7,
        }
    }

    /// Reset encoder state for a new chunk.
    pub fn reset(&mut self) {
        self.low = 0;
        self.high = STATE_MASK;
        self.underflow = 0;
        self.output.clear();
        self.current_byte = 0;
        self.bit_pos = 7;
    }

    /// Emit a single bit to the output stream.
    #[inline]
    fn emit_bit(&mut self, bit: u8) {
        if bit != 0 {
            self.current_byte |= 1 << self.bit_pos;
        }
        self.bit_pos -= 1;

        if self.bit_pos < 0 {
            self.output.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 7;
        }
    }

    /// Emit a bit and all pending underflow bits.
    #[inline]
    fn emit_bit_plus_underflow(&mut self, bit: u8) {
        self.emit_bit(bit);
        let opposite = bit ^ 1;
        for _ in 0..self.underflow {
            self.emit_bit(opposite);
        }
        self.underflow = 0;
    }

    /// Encode a symbol given its cumulative frequency range.
    ///
    /// Arguments:
    /// - cumfreqs: cumulative frequencies array (inclusive prefix sum)
    /// - symbol: the symbol to encode (0-indexed)
    pub fn encode_symbol(&mut self, cumfreqs: &[u32], symbol: usize) {
        let total = cumfreqs[cumfreqs.len() - 1] as u64;

        // Get symbol's frequency range
        let sym_high = cumfreqs[symbol] as u64;
        let sym_low = if symbol > 0 {
            cumfreqs[symbol - 1] as u64
        } else {
            0
        };

        // Update interval
        let range = self.high - self.low + 1;
        self.high = self.low + (sym_high * range / total) - 1;
        self.low = self.low + (sym_low * range / total);

        // Renormalize
        self.renormalize();
    }

    /// Renormalization loop (E1, E2, E3 conditions).
    fn renormalize(&mut self) {
        loop {
            // E1/E2: MSBs match - shift out
            if (self.low ^ self.high) & HALF_RANGE == 0 {
                let out_bit = ((self.low >> (NUM_STATE_BITS - 1)) & 1) as u8;
                self.emit_bit_plus_underflow(out_bit);
                self.low = (self.low << 1) & STATE_MASK;
                self.high = ((self.high << 1) & STATE_MASK) | 1;
            }
            // E3: Underflow condition
            else if (self.low & !self.high & QUARTER_RANGE) != 0 {
                self.underflow += 1;
                self.low = ((self.low << 1) ^ HALF_RANGE) & STATE_MASK;
                self.high = (((self.high ^ HALF_RANGE) << 1) | HALF_RANGE | 1) & STATE_MASK;
            } else {
                break;
            }
        }
    }

    /// Finish encoding and flush remaining bits.
    pub fn finish(&mut self) {
        // Emit terminating bit
        self.emit_bit(1);

        // Flush partial byte
        if self.bit_pos < 7 {
            self.output.push(self.current_byte);
        }
    }

    /// Get the encoded bitstream.
    pub fn get_output(&self) -> &[u8] {
        &self.output
    }
}

/// Arithmetic decoder state.
pub struct ArithDecoder {
    pub low: u64,
    pub high: u64,
    pub code: u64,
    /// Input bitstream
    input: Vec<u8>,
    /// Current byte index
    byte_pos: usize,
    /// Bit position in current byte (7 = MSB, 0 = LSB)
    bit_pos: i32,
}

impl ArithDecoder {
    /// Create a new decoder with the given input.
    pub fn new(input: Vec<u8>) -> Self {
        let mut decoder = Self {
            low: 0,
            high: STATE_MASK,
            code: 0,
            input,
            byte_pos: 0,
            bit_pos: 7,
        };
        decoder.init_code();
        decoder
    }

    /// Initialize code register from first 32 bits.
    fn init_code(&mut self) {
        self.code = 0;
        for _ in 0..NUM_STATE_BITS {
            self.code = (self.code << 1) | self.read_bit() as u64;
        }
    }

    /// Read a single bit from the input stream.
    #[inline]
    fn read_bit(&mut self) -> u8 {
        if self.byte_pos >= self.input.len() {
            return 0; // Pad with zeros
        }

        let bit = (self.input[self.byte_pos] >> self.bit_pos) & 1;
        self.bit_pos -= 1;

        if self.bit_pos < 0 {
            self.byte_pos += 1;
            self.bit_pos = 7;
        }

        bit
    }

    /// Decode the next symbol given cumulative frequencies.
    ///
    /// Returns the decoded symbol (0-indexed).
    pub fn decode_symbol(&mut self, cumfreqs: &[u32]) -> usize {
        let total = cumfreqs[cumfreqs.len() - 1] as u64;
        let range = self.high - self.low + 1;
        let offset = self.code - self.low;

        // Compute value in [0, total)
        let value = ((offset + 1) * total - 1) / range;

        // Binary search for symbol
        let symbol = self.bisect_cumfreqs(cumfreqs, value as u32);

        // Get symbol's frequency range
        let sym_high = cumfreqs[symbol] as u64;
        let sym_low = if symbol > 0 {
            cumfreqs[symbol - 1] as u64
        } else {
            0
        };

        // Update interval (same as encoder)
        self.high = self.low + (sym_high * range / total) - 1;
        self.low = self.low + (sym_low * range / total);

        // Renormalize
        self.renormalize();

        symbol
    }

    /// Binary search to find symbol from cumfreqs.
    fn bisect_cumfreqs(&self, cumfreqs: &[u32], value: u32) -> usize {
        // Find smallest i such that value < cumfreqs[i]
        let mut lo = 0;
        let mut hi = cumfreqs.len();

        while lo < hi {
            let mid = (lo + hi) / 2;
            if value < cumfreqs[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        // Clamp to valid symbol range
        lo.min(cumfreqs.len() - 1)
    }

    /// Renormalization loop for decoder.
    fn renormalize(&mut self) {
        loop {
            if (self.low ^ self.high) & HALF_RANGE == 0 {
                // E1/E2: MSBs match - shift
                self.low = (self.low << 1) & STATE_MASK;
                self.high = ((self.high << 1) & STATE_MASK) | 1;
                self.code = ((self.code << 1) & STATE_MASK) | self.read_bit() as u64;
            } else if (self.low & !self.high & QUARTER_RANGE) != 0 {
                // E3: Underflow - matches CUDA arith_cuda.cu
                self.low = ((self.low << 1) ^ HALF_RANGE) & STATE_MASK;
                self.high = (((self.high ^ HALF_RANGE) << 1) | HALF_RANGE | 1) & STATE_MASK;
                // CUDA: code = (code & HALF_RANGE) | ((code << 1) & (STATE_MASK >> 1)) | bit
                // This preserves the MSB, shifts rest left, and adds new bit
                let half_bit = self.code & HALF_RANGE;
                let shifted = (self.code << 1) & (STATE_MASK >> 1);
                self.code = half_bit | shifted | self.read_bit() as u64;
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_cumfreqs(n: usize) -> Vec<u32> {
        // Uniform distribution: each symbol has equal frequency
        let freq_each = 1000u32;
        (1..=n).map(|i| i as u32 * freq_each).collect()
    }

    #[test]
    fn test_encode_decode_single() {
        let cumfreqs = make_uniform_cumfreqs(4);

        let mut encoder = ArithEncoder::new();
        encoder.encode_symbol(&cumfreqs, 2);
        encoder.finish();

        let encoded = encoder.get_output().to_vec();
        let mut decoder = ArithDecoder::new(encoded);
        let decoded = decoder.decode_symbol(&cumfreqs);

        assert_eq!(decoded, 2);
    }

    #[test]
    fn test_encode_decode_sequence() {
        let cumfreqs = make_uniform_cumfreqs(8);
        let symbols = vec![0, 3, 7, 2, 5, 1];

        let mut encoder = ArithEncoder::new();
        for &sym in &symbols {
            encoder.encode_symbol(&cumfreqs, sym);
        }
        encoder.finish();

        let encoded = encoder.get_output().to_vec();
        let mut decoder = ArithDecoder::new(encoded);

        for &expected in &symbols {
            let decoded = decoder.decode_symbol(&cumfreqs);
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn test_encode_decode_longer_sequence() {
        // Test with a longer sequence to verify stability
        let cumfreqs = make_uniform_cumfreqs(8);
        let symbols: Vec<usize> = (0..16).map(|i| i % 8).collect();

        let mut encoder = ArithEncoder::new();
        for &sym in &symbols {
            encoder.encode_symbol(&cumfreqs, sym);
        }
        encoder.finish();

        let encoded = encoder.get_output().to_vec();
        let mut decoder = ArithDecoder::new(encoded);

        for &expected in &symbols {
            let decoded = decoder.decode_symbol(&cumfreqs);
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn test_reset() {
        let cumfreqs = make_uniform_cumfreqs(4);

        let mut encoder = ArithEncoder::new();
        encoder.encode_symbol(&cumfreqs, 1);
        encoder.finish();

        encoder.reset();
        encoder.encode_symbol(&cumfreqs, 2);
        encoder.finish();

        let encoded = encoder.get_output().to_vec();
        let mut decoder = ArithDecoder::new(encoded);
        assert_eq!(decoder.decode_symbol(&cumfreqs), 2);
    }
}
