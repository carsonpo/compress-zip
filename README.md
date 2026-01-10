# compress-zip

Deterministic CPU implementations of a neural text compression system in **Rust** and **Python**.

These implementations are designed to be **bit-exact verifiers** for the GPU (CUDA) implementation, enabling cross-platform validation of compressed files.

## Overview

This repository contains reference implementations of the DETAC (Deterministic Text Arithmetic Coding) neural compression system. The core idea is to use a small language model to predict token probabilities, then encode tokens using arithmetic coding based on those probabilities.

### Key Features

- **Bit-exact determinism**: All operations use integer arithmetic with ties-to-even rounding
- **Cross-language verification**: Python and Rust implementations produce identical outputs
- **No floating point in hot path**: Uses Q-format fixed-point throughout (Q0.7 for activations, Q1.15 for RoPE, Q16 for exp)
- **Minimal dependencies**: Self-contained implementations without heavy ML frameworks

## Compression Benchmarks

Comparison on Alice in Wonderland Chapter 1 (11,782 bytes):

| Method | Size | Ratio | vs Best Traditional |
|--------|------|-------|---------------------|
| **CZIP (Rust)** | 3,952 | **2.98x** | baseline |
| brotli-11 | 4,251 | 2.77x | +7.6% larger |
| zstd-22 | 4,841 | 2.43x | +22.5% larger |
| gzip-9 | 5,011 | 2.35x | +26.8% larger |

**Speed**: ~27,000 tokens/sec encode/decode (parallel, Rust with rayon)

**System**: AMD EPYC 9354 32-Core Processor, 1.1 TiB RAM

### Comparison with Other Neural Compressors

| Compressor | Model | Compression | Speed | Hardware |
|------------|-------|-------------|-------|----------|
| **CZIP** | 1-layer, 1024 vocab | 2.98x | ~27,000 tok/s | CPU |
| [ts_zip](https://bellard.org/ts_zip/) | RWKV 169M | ~7x | ~577 tok/s | RTX 4090 GPU |
| [llama-zip](https://github.com/alexbuz/llama-zip) | Llama 3.1 8B | 8-29x | ~30 tok/s | GPU required |


### Architecture

```
Input Text
    │
    ▼
┌─────────────┐
│  Tokenizer  │  (tiktoken BPE)
└─────────────┘
    │ tokens
    ▼
┌─────────────┐
│  LM Model   │  (int8 weights, int32 accumulators)
│  - Embed    │
│  - Attn+RoPE│
│  - FFN+ReGLU│
│  - RMSNorm  │
└─────────────┘
    │ logits (int32)
    ▼
┌─────────────┐
│ Softmax CDF │  (integer exp2 LUT)
└─────────────┘
    │ cumfreqs
    ▼
┌─────────────┐
│ Arith Coder │  (32-bit state, E1/E2/E3 renorm)
└─────────────┘
    │
    ▼
Compressed Bytes (.czip)
```

## Project Structure

```
compress-zip/
├── rust/                    # Rust implementation
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Module exports
│       ├── primitives.rs    # Deterministic integer ops
│       ├── lut.rs           # Exp2 and RoPE lookup tables
│       ├── rmsnorm.rs       # Integer RMSNorm
│       ├── reglu.rs         # ReGLU activation
│       ├── linear.rs        # Int8 GEMM
│       ├── attention.rs     # GQA/MQA with RoPE
│       ├── softmax_cdf.rs   # Logits → CDF conversion
│       ├── arith_coder.rs   # Arithmetic encoder/decoder
│       ├── file_format.rs   # DETACv1 file format
│       ├── safetensors.rs   # Model weight loader
│       └── tiktoken.rs      # BPE tokenizer
│
├── python/                  # Python implementation
│   ├── pyproject.toml
│   ├── compress_zip/        # Main package
│   │   ├── primitives.py
│   │   ├── lut.py
│   │   ├── rmsnorm.py
│   │   ├── reglu.py
│   │   ├── linear.py
│   │   ├── attention.py
│   │   ├── softmax_cdf.py
│   │   ├── arith_coder.py
│   │   └── file_format.py
│   └── tests/               # Unit tests
│       ├── test_primitives.py
│       ├── test_lut.py
│       └── test_arith_coder.py
│
└── README.md
```

## Installation

### Rust

```bash
cd rust
cargo build --release
```

### Python

```bash
cd python
uv pip install -e ".[dev]"
# or with pip:
pip install -e ".[dev]"
```

## Running Tests

### Rust Tests

```bash
cd rust
cargo test
```

Expected output: `40 passed`

### Python Tests

```bash
cd python
pytest tests/ -v
```

Expected output: `66 passed`

### Run All Tests

```bash
# From repository root
(cd rust && cargo test) && (cd python && pytest tests/ -v)
```

## Cross-Language Verification

The implementations include cross-verification test vectors to ensure Python and Rust produce identical results.

### Primitive Operations

Both implementations must produce the same output for these operations:

| Operation | Input | Expected |
|-----------|-------|----------|
| `sra_rne_tte_s32(7, 1)` | 7 >> 1 with ties-to-even | 4 |
| `sra_rne_tte_s32(5, 1)` | 5 >> 1 with ties-to-even | 2 |
| `isqrt32_restoring(1000000)` | √1000000 | 1000 |
| `udiv_rne_tte_u32(1000000, 3)` | 1000000 ÷ 3 rounded | 333334 |

### Verifying Cross-Language Consistency

```python
# Python
from compress_zip import sra_rne_tte_s32, isqrt32_restoring
print(sra_rne_tte_s32(1234567, 10))  # Should print: 1206
print(isqrt32_restoring(2147483647))  # Should print: 46340
```

```rust
// Rust
use compress_zip::primitives::{sra_rne_tte_s32, isqrt32_restoring};
println!("{}", sra_rne_tte_s32(1234567, 10));  // Should print: 1206
println!("{}", isqrt32_restoring(2147483647)); // Should print: 46340
```

### Arithmetic Coder Verification

```python
# Python - encode and decode roundtrip
from compress_zip import ArithEncoder, ArithDecoder
import numpy as np

# Create frequency table
freqs = np.array([1000, 1000, 1000, 1000], dtype=np.int32)
cumfreqs = np.zeros(5, dtype=np.int32)
cumfreqs[1:] = np.cumsum(freqs)
total = int(cumfreqs[-1])

# Encode
symbols = [0, 1, 2, 3, 0, 1, 2, 3]
encoder = ArithEncoder()
for sym in symbols:
    encoder.encode_symbol(cumfreqs, sym, total)
compressed = encoder.finish()

# Decode
decoder = ArithDecoder(compressed)
decoded = [decoder.decode_symbol(cumfreqs, total) for _ in symbols]

assert symbols == decoded
print(f"Compressed {len(symbols)} symbols to {len(compressed)} bytes")
```

## File Format: CZIPv1

Compressed files use the CZIPv1 format with a two-layer structure:

### Outer Envelope (16 bytes + compressed payload)
```
┌────────────────────────────────────┐
│ magic: 6 bytes "CZIPv1"            │
│ flags: 1 byte                      │
│   bit0: is_multifile               │
│   bit1: training_marker            │
│   bit2: reserved                   │
│ codec_id: 1 byte (0=zstd, 1=brotli)│
│ uncompressed_len: u32 LE           │
│   (original text byte length)      │
│ header_len: u32 LE                 │
│   (inner header size)              │
├────────────────────────────────────┤
│ compressed payload                 │
│   (zstd level 22 or brotli 11)     │
└────────────────────────────────────┘
```

### Inner Header (variable size, before compression)
```
┌────────────────────────────────────┐
│ model_id_hash: u32 LE              │
│ chunk_count: u32 LE                │
│ last_chunk_tokens: u16 LE (1..64)  │
│ tokenizer_id: u8                   │
│ reserved: u8                       │
│ chunk_byte_lens: [u16; chunk_count]│
├────────────────────────────────────┤
│ chunk_data: concatenated chunks    │
└────────────────────────────────────┘
```

All chunks except the last contain exactly 64 tokens. The last chunk can have 1-64 tokens as specified by `last_chunk_tokens`.

## Compression Ratio Testing

### Python Example

```python
from compress_zip import (
    ArithEncoder, ArithDecoder,
    build_cumfreqs, CompressedFile
)
import numpy as np

def estimate_compression_ratio(tokens, logits_per_token):
    """
    Estimate compression ratio given tokens and their predicted logits.

    Args:
        tokens: List of token IDs
        logits_per_token: List of logit arrays, one per token position

    Returns:
        Compression ratio (original_bits / compressed_bits)
    """
    encoder = ArithEncoder()
    vocab_size = len(logits_per_token[0])

    for i, token in enumerate(tokens[1:], 1):  # Skip BOS
        logits = logits_per_token[i - 1]
        cumfreqs, total = build_cumfreqs(logits, vocab_size)
        encoder.encode_symbol(cumfreqs, token, total)

    compressed = encoder.finish()

    # Original: log2(vocab_size) bits per token
    original_bits = len(tokens) * np.log2(vocab_size)
    compressed_bits = len(compressed) * 8

    return original_bits / compressed_bits

# Example with random logits (for testing only - real model would give better ratios)
vocab_size = 1024
num_tokens = 100
tokens = np.random.randint(0, vocab_size, num_tokens)
logits = [np.random.randint(-1000, 1000, vocab_size, dtype=np.int32)
          for _ in range(num_tokens)]

ratio = estimate_compression_ratio(tokens, logits)
print(f"Compression ratio: {ratio:.2f}x")
```

### Decompression Verification

```python
def verify_roundtrip(tokens, logits_per_token):
    """Verify that encode → decode produces original tokens."""
    vocab_size = len(logits_per_token[0])

    # Encode
    encoder = ArithEncoder()
    cumfreqs_list = []
    for i, token in enumerate(tokens[1:], 1):
        logits = logits_per_token[i - 1]
        cumfreqs, total = build_cumfreqs(logits, vocab_size)
        cumfreqs_list.append((cumfreqs, total))
        encoder.encode_symbol(cumfreqs, token, total)

    compressed = encoder.finish()

    # Decode
    decoder = ArithDecoder(compressed)
    decoded_tokens = [tokens[0]]  # Start with BOS

    for cumfreqs, total in cumfreqs_list:
        token = decoder.decode_symbol(cumfreqs, total)
        decoded_tokens.append(token)

    # Verify
    assert tokens.tolist() == decoded_tokens, "Roundtrip failed!"
    print(f"Verified {len(tokens)} tokens, {len(compressed)} bytes")
    return True
```

### Slow Compression Demo

To see compression in action with detailed output:

```python
import numpy as np
from compress_zip import ArithEncoder, build_cumfreqs

def slow_compress_demo(text_tokens, model_forward_fn, vocab_size=1024):
    """
    Demonstrate compression with step-by-step output.

    Args:
        text_tokens: Token IDs to compress
        model_forward_fn: Function that returns logits given context
        vocab_size: Size of vocabulary
    """
    encoder = ArithEncoder()
    total_bits = 0

    print(f"Compressing {len(text_tokens)} tokens...")
    print("-" * 60)

    context = [text_tokens[0]]  # Start with BOS

    for i, token in enumerate(text_tokens[1:], 1):
        # Get model predictions
        logits = model_forward_fn(context)
        cumfreqs, total = build_cumfreqs(logits, vocab_size)

        # Compute information content
        freq = cumfreqs[token + 1] - cumfreqs[token]
        bits = -np.log2(freq / total)
        total_bits += bits

        # Encode
        encoder.encode_symbol(cumfreqs, token, total)

        # Update context
        context.append(token)

        if i <= 10 or i % 50 == 0:
            print(f"Token {i:4d}: id={token:4d}, freq={freq:6d}/{total}, "
                  f"bits={bits:.2f}, cumulative={total_bits:.1f}")

    compressed = encoder.finish()

    print("-" * 60)
    print(f"Total tokens:     {len(text_tokens)}")
    print(f"Compressed bytes: {len(compressed)}")
    print(f"Bits per token:   {len(compressed) * 8 / len(text_tokens):.2f}")
    print(f"Entropy estimate: {total_bits / len(text_tokens):.2f} bits/token")

    # Compare to naive encoding
    naive_bits = len(text_tokens) * np.log2(vocab_size)
    print(f"Naive encoding:   {naive_bits / 8:.0f} bytes")
    print(f"Compression ratio: {naive_bits / (len(compressed) * 8):.2f}x")

    return compressed
```

## Constants and Configuration

### Quantization Format

| Type | Format | Scale | Range |
|------|--------|-------|-------|
| Activations | Q0.7 | 128 | [-1, 0.992] |
| RoPE cos/sin | Q1.15 | 32768 | [-1, 1) |
| Exp2 LUT | Q16 | 65536 | [0, 1] |
| Logits | int32 | - | full range |

### Arithmetic Coder

```
NUM_STATE_BITS = 32
HALF_RANGE     = 2^31  (0x80000000)
QUARTER_RANGE  = 2^30  (0x40000000)
```

### Attention

```
HEAD_DIM       = 64
MAX_SEQ_LEN    = 64
TOKENS_PER_BATCH = 16
n_kv_heads     = 1  (Multi-Query Attention)
```

## Troubleshooting

### Common Issues

1. **Python import errors**: Make sure to install with `pip install -e .`

2. **Rust compilation errors**: Ensure you have Rust 1.70+ installed

3. **Numerical differences**: Both implementations use ties-to-even rounding. If you see differences, check:
   - Shift amounts in `sra_rne_tte_s32`
   - Integer overflow (use 64-bit intermediates)
   - LUT indexing (exp2 uses 8-bit fractional index)

### Debugging Tips

```python
# Enable verbose output in Python tests
pytest tests/ -v -s

# Run specific test
pytest tests/test_primitives.py::TestSraRneTteS32 -v
```

```bash
# Run specific Rust test with output
cargo test test_sra_rne_tte -- --nocapture
```

## API Reference

### Python

```python
# Primitives
sra_rne_tte_s32(x: int, shift: int) -> int
isqrt32_restoring(x: int) -> int
udiv_rne_tte_u32(num: int, div: int) -> int
clamp_i8(x: int) -> int

# LUTs
Exp2LutQ16()           # 256-entry exp2 fractional table
RopeLut()              # cos/sin tables for rotary embeddings

# Layers
rmsnorm_i8(x, weight)  # Integer RMSNorm
reglu_i8(gate, value)  # ReGLU activation
linear_i8_to_i32(x, weight, bias=None)  # Int8 GEMM

# Attention
gqa_attention_mqa_i8(q, k, v, pos, kv_cache=None)
KVCache.create(max_seq_len, head_dim)

# Coding
build_cumfreqs(logits, vocab_size) -> (cumfreqs, total)
ArithEncoder() / ArithDecoder(data)

# File I/O
CZIPv1OuterHeader / CZIPv1InnerHeader
CompressedFile.create(...) / CompressedFile.read(path)
```

### Rust

```rust
use compress_zip::{
    primitives::{sra_rne_tte_s32, isqrt32_restoring},
    lut::{Exp2LutQ16, RopeLut},
    rmsnorm::rmsnorm_i8,
    attention::{gqa_attention_mqa_i8, KVCache},
    softmax_cdf::build_cumfreqs,
    arith_coder::{ArithEncoder, ArithDecoder},
    file_format::{CZIPv1OuterHeader, CZIPv1InnerHeader, CompressedFile, Codec},
};
```

## License

See repository root for license information.

## References

- Arithmetic coding: Witten, Neal, Cleary (1987)
- RoPE: Su et al. "RoFormer" (2021)
- Ties-to-even rounding: IEEE 754 standard
