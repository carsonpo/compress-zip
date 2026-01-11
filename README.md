# compress-zip

CPU reference implementations (Rust and Python) for the [compress.zip](https://compress.zip) neural compression API.

## What is this?

This repo lets you independently verify compression ratios and determinism of data compressed through the compress.zip API. Both implementations produce bit-exact identical output, so you can validate that your compressed files decompress correctly without relying on our servers.

The compress.zip service lets you upload a text corpus, trains a bespoke tokenizer and compression model for your data, and then compresses/decompresses at 30-50 MB/s on GPU. You can export your trained model and use it with this repo for offline verification.

**Note:** The compress.zip service is currently in development. If you are interested in many orders of magnitude faster performance than this repo, feel free to email me at carson at poole.ai

## Compression Performance

Wikipedia articles (~15KB each):

| Article | Original | CZIP | brotli-11 | zstd-22 | gzip-9 |
|---------|----------|------|-----------|---------|--------|
| Machine Learning | 15,003 | **3,876 (3.87x)** | 4,303 (3.49x) | 5,541 (2.71x) | 5,783 (2.59x) |
| World War II | 15,024 | **4,316 (3.48x)** | 4,729 (3.18x) | 5,914 (2.54x) | 6,139 (2.45x) |
| Python (PL) | 15,020 | **4,336 (3.46x)** | 4,670 (3.22x) | 5,852 (2.57x) | 6,079 (2.47x) |
| Einstein | 15,038 | **4,392 (3.42x)** | 4,920 (3.06x) | 6,306 (2.38x) | 6,546 (2.30x) |
| Rust (PL) | 15,014 | **4,447 (3.38x)** | 4,727 (3.18x) | 6,001 (2.50x) | 6,217 (2.41x) |

### Comparison with Other Neural Compressors

| Compressor | Model | Compression | Speed | Hardware |
|------------|-------|-------------|-------|----------|
| **CZIP** | tiny GPT model | 3.4-3.9x | ~28,000 tok/s | CPU |
| [ts_zip](https://bellard.org/ts_zip/) | RWKV 169M | ~7x | ~577 tok/s | RTX 4090 GPU |
| [llama-zip](https://github.com/alexbuz/llama-zip) | Llama 3.1 8B | 8-29x | ~30 tok/s | GPU |


## How it works

Zero floating-point operations in the entire inference path. Every computation—matrix multiplies, attention scores, softmax, normalization—is pure integer math. Transcendental functions (exp, sin, cos for rotary embeddings) come from precomputed lookup tables, not FPU instructions.

This means you get bit-exact results on any hardware: x86, ARM, a microcontroller, a toaster with a CPU. No floating-point rounding differences, no platform-specific math libraries, no surprises.

- **Weights**: int8
- **Activations**: Q0.7 fixed-point
- **Accumulators**: int32
- **RMSNorm**: integer square root
- **Softmax**: exp2 LUT (Q16 output)
- **RoPE**: Q1.15 sin/cos tables
- **Arithmetic coder**: 32-bit state, E1/E2/E3 renormalization

## Installation

### Rust

```bash
cd rust
cargo build --release
```

### Python

```bash
cd python
pip install -e ".[dev]"
```

## Usage

### Compress

```bash
# Rust
./target/release/compress_zip compress -m models/v000_eng.czm -i input.txt -o output.czip

# Python
python -m compress_zip compress -m models/v000_eng.czm -i input.txt -o output.czip
```

### Decompress

```bash
# Rust
./target/release/compress_zip decompress -m models/v000_eng.czm -i output.czip -o recovered.txt

# Python
python -m compress_zip decompress -m models/v000_eng.czm -i output.czip -o recovered.txt
```

### Options

- `-m, --model` — Path to model file (safetensors format)
- `-t, --tokenizer` — Tokenizer path (default: uses tokenizer embedded in model)
- `-c, --codec` — Outer compression: `brotli` (default) or `zstd`

## Running Tests

```bash
cd rust && cargo test

cd python && pytest tests/ -v
```

## License
MIT