# compress-zip

Deterministic CPU implementation of neural compression.

## Overview

This package provides bit-exact CPU implementations of the neural compression
algorithms, designed to match the CUDA GPU implementations for verification.

## Features

- **Deterministic primitives**: Integer arithmetic with ties-to-even rounding
- **Integer RMSNorm**: Layer normalization using integer arithmetic
- **Integer attention**: GQA/MQA attention with RoPE in integer math
- **Arithmetic coding**: Range-based entropy coding for compression
- **DETACv1 format**: File format for compressed data

## Installation

```bash
uv pip install -e .
```

## Usage

```python
from compress_zip import (
    sra_rne_tte_s32,
    rmsnorm_i8,
    gqa_attention_mqa_i8,
    ArithEncoder,
    ArithDecoder,
)
```

## Testing

```bash
pytest tests/
```
