"""
Export model to czip-model-v1 format with embedded LUTs and tokenizer.

This script creates a self-contained safetensors file that includes:
- All model weights (int8/int16 with shift-multiply parameters)
- Exp2 LUT (256 x uint16) - matches CUDA softmax_cumfreq.cu exactly
- RoPE LUT (cos/sin in Q1.15) - matches CUDA attention.cu exactly
- Tokenizer (as JSON blob in metadata)

The LUTs are generated here and baked into the model to ensure bit-exactness
across all implementations (Python, Rust, CUDA).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Constants matching CUDA
EXP_FRAC_SIZE = 256  # 2^8 entries
ROPE_MAX_SEQ_LEN = 64  # Default max sequence length
ROPE_HEAD_DIM = 64
ROPE_HALF_DIM = ROPE_HEAD_DIM // 2
ROPE_BASE = 10000.0


def generate_exp2_lut_cuda() -> np.ndarray:
    """
    Generate exp2 LUT matching CUDA softmax_cumfreq.cu exactly.

    CUDA code:
        for (int i = 0; i < EXP_FRAC_SIZE; ++i) {
            double frac = (double)i / (double)EXP_FRAC_SIZE;
            double v = std::pow(2.0, -frac) * 65536.0;  // NEGATIVE exponent
            int64_t q = round_ties_to_even_host(v);
            if (q < 1) q = 1;
            if (q > 65535) q = 65535;
            host[i] = (uint16_t)q;
        }
    """
    table = np.zeros(EXP_FRAC_SIZE, dtype=np.uint16)
    for i in range(EXP_FRAC_SIZE):
        frac = i / EXP_FRAC_SIZE
        v = math.pow(2.0, -frac) * 65536.0  # NEGATIVE exponent
        # Python 3's round() uses banker's rounding (ties-to-even) like llrint
        q = round(v)
        q = max(1, min(65535, int(q)))
        table[i] = q
    return table


def generate_rope_lut_cuda(
    max_seq_len: int = ROPE_MAX_SEQ_LEN,
    head_dim: int = ROPE_HEAD_DIM,
    base: float = ROPE_BASE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate RoPE LUT matching CUDA/Python implementation.

    theta_i = pos * (1.0 / base^(2*i / head_dim))
    cos[pos,i] = clamp(round(cos(theta_i) * 32768), -32768, 32767)
    sin[pos,i] = clamp(round(sin(theta_i) * 32768), -32768, 32767)

    Returns (cos_lut, sin_lut) each of shape [max_seq_len, half_dim]
    """
    half_dim = head_dim // 2
    cos_lut = np.zeros((max_seq_len, half_dim), dtype=np.int16)
    sin_lut = np.zeros((max_seq_len, half_dim), dtype=np.int16)

    for pos in range(max_seq_len):
        for i in range(half_dim):
            # Frequency: 1 / (base ^ (2*i / head_dim))
            freq = 1.0 / math.pow(base, (2 * i) / head_dim)
            theta = pos * freq

            # Q1.15 format with ties-to-even rounding
            cos_val = round(math.cos(theta) * 32768.0)
            sin_val = round(math.sin(theta) * 32768.0)

            # Clamp to i16 range
            cos_lut[pos, i] = max(-32768, min(32767, int(cos_val)))
            sin_lut[pos, i] = max(-32768, min(32767, int(sin_val)))

    return cos_lut, sin_lut


def export_model(
    output_path: str,
    weights: Dict[str, np.ndarray],
    config: Dict[str, Any],
    tokenizer_json: Optional[str] = None,
    max_seq_len: int = ROPE_MAX_SEQ_LEN,
    head_dim: int = ROPE_HEAD_DIM,
    rope_base: float = ROPE_BASE,
):
    """
    Export model to czip-model-v1 safetensors format.

    Args:
        output_path: Path to output .safetensors file
        weights: Dictionary of weight tensors (numpy arrays)
        config: Model configuration dictionary
        tokenizer_json: Path to tokenizer JSON file (optional)
        max_seq_len: Maximum sequence length for RoPE LUT
        head_dim: Head dimension for RoPE LUT
        rope_base: Base for RoPE frequency calculation
    """
    try:
        from safetensors.numpy import save_file
    except ImportError:
        print("Error: safetensors not installed. Run: pip install safetensors")
        return

    # Generate LUTs
    print("Generating LUTs...")
    exp2_lut = generate_exp2_lut_cuda()
    rope_cos, rope_sin = generate_rope_lut_cuda(max_seq_len, head_dim, rope_base)

    # Add LUTs to weights
    tensors = dict(weights)
    tensors["lut.exp2"] = exp2_lut  # [256] uint16
    tensors["lut.rope_cos"] = rope_cos  # [max_seq_len, half_dim] int16
    tensors["lut.rope_sin"] = rope_sin  # [max_seq_len, half_dim] int16

    # Prepare metadata
    metadata = {
        "format": "czip-model-v1",
        "config": json.dumps(config),
    }

    # Add tokenizer if provided
    if tokenizer_json:
        print(f"Loading tokenizer from {tokenizer_json}...")
        with open(tokenizer_json, 'r') as f:
            tokenizer_data = f.read()
        metadata["tokenizer"] = tokenizer_data

    # Add LUT config to metadata
    lut_config = {
        "exp2_size": EXP_FRAC_SIZE,
        "rope_max_seq_len": max_seq_len,
        "rope_head_dim": head_dim,
        "rope_base": rope_base,
    }
    metadata["lut_config"] = json.dumps(lut_config)

    # Save
    print(f"Saving to {output_path}...")
    save_file(tensors, output_path, metadata=metadata)
    print(f"Exported {len(tensors)} tensors")
    print(f"  - exp2_lut: {exp2_lut.shape} uint16")
    print(f"  - rope_cos: {rope_cos.shape} int16")
    print(f"  - rope_sin: {rope_sin.shape} int16")
    if tokenizer_json:
        print(f"  - tokenizer: {len(tokenizer_data)} bytes")


def verify_luts():
    """Verify LUT generation matches expected values."""
    print("Verifying LUT generation...")

    # Verify exp2 LUT
    exp2_lut = generate_exp2_lut_cuda()
    assert exp2_lut[0] == 65535, f"exp2[0] = {exp2_lut[0]}, expected 65535"
    # 2^(-0.5) * 65536 ≈ 46341
    expected_half = round(math.pow(2, -0.5) * 65536)
    assert abs(exp2_lut[128] - expected_half) <= 1
    # Table should be monotonically decreasing
    for i in range(1, EXP_FRAC_SIZE):
        assert exp2_lut[i] <= exp2_lut[i-1], f"exp2 not decreasing at {i}"
    print("  exp2_lut: OK")

    # Verify RoPE LUT
    rope_cos, rope_sin = generate_rope_lut_cuda()
    # At pos=0, all angles are 0: cos(0)=1, sin(0)=0
    assert rope_cos[0, 0] == 32767, f"rope_cos[0,0] = {rope_cos[0, 0]}, expected 32767"
    assert rope_sin[0, 0] == 0, f"rope_sin[0,0] = {rope_sin[0, 0]}, expected 0"
    print("  rope_lut: OK")

    print("LUT verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Export model to czip-model-v1 format")
    parser.add_argument("--verify", action="store_true", help="Verify LUT generation")
    parser.add_argument("--output", "-o", help="Output safetensors path")
    parser.add_argument("--tokenizer", "-t", help="Path to tokenizer.json")
    parser.add_argument("--checkpoint", "-c", help="Path to source checkpoint")
    args = parser.parse_args()

    if args.verify:
        verify_luts()
        return

    if not args.output:
        print("Error: --output required")
        return

    # For now, just demonstrate LUT export
    print("Generating standalone LUT export...")

    # Create minimal config
    config = {
        "vocab_size": 1024,
        "n_layers": 1,
        "n_heads": 6,
        "d_model": 384,
        "d_ff": 1024,
        "head_dim": 64,
        "max_seq_len": 64,
    }

    # Empty weights for demo (would come from checkpoint)
    weights = {}

    export_model(
        args.output,
        weights,
        config,
        tokenizer_json=args.tokenizer,
    )


if __name__ == "__main__":
    main()
