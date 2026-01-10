"""
Neural compression model implementation.

Small transformer model for text compression using int8 arithmetic.
Supports czip-model-v1 format (integer-only: pre-quantized int8 weights with
shift-multiply requantization and int16 Q1.14 norm weights).
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .attention import gqa_attention_mqa_i8, gqa_attention_mqa_i8_cached, apply_rope_i8, KVCache, HEAD_DIM, MAX_SEQ_LEN
from .linear import linear_i8_to_i32
from .lut import get_exp2_lut, get_rope_lut
from .primitives import clamp_i8, sra_rne_tte_s32
from .reglu import reglu_i8
from .rmsnorm import rmsnorm_i8_int
from .softmax_cdf import build_cumfreqs

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False


@dataclass
class ModelConfig:
    """Model configuration."""
    vocab_size: int = 1024
    n_layers: int = 1
    n_heads: int = 6
    n_kv_heads: int = 1
    d_model: int = 384
    d_ff: int = 1024
    head_dim: int = 64
    max_seq_len: int = 8192
    rope_theta_q16: int = 655360000  # 10000.0 * 65536

    @classmethod
    def from_metadata(cls, metadata: Dict[str, str]) -> "ModelConfig":
        """Load config from safetensors metadata."""
        def get_val(key: str, default: Any) -> Any:
            if key in metadata:
                val = metadata[key]
                # Metadata values are JSON strings
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return val
            return default

        return cls(
            vocab_size=get_val("vocab_size", 1024),
            n_layers=get_val("n_layers", 1),
            n_heads=get_val("n_heads", 6),
            n_kv_heads=get_val("n_kv_heads", 1),
            d_model=get_val("d_model", 384),
            d_ff=get_val("d_ff", 1024),
            head_dim=get_val("head_dim", 64),
            max_seq_len=get_val("max_seq_len", 8192),
            rope_theta_q16=get_val("rope_theta_q16", 655360000),
        )


@dataclass
class LayerWeights:
    """Transformer layer weights (czip-model-v1 integer-only format)."""
    # Pre-attention norm (Q1.14)
    attn_norm: np.ndarray       # [d_model] int16

    # Attention
    qkv_weight: np.ndarray      # [d_model + 2*kv_dim, d_model] int8
    qkv_mult: np.int32          # scalar
    qkv_shift: np.int8          # scalar
    out_weight: np.ndarray      # [d_model, d_model] int8
    out_mult: np.int32          # scalar
    out_shift: np.int8          # scalar
    score_mult: np.ndarray      # [n_heads] int16 (Q0.15)
    score_shift: np.int8        # scalar

    # Post-attention norm (Q1.14)
    post_attn_norm: np.ndarray  # [d_model] int16

    # Pre-FFN norm (Q1.14)
    ffn_norm: np.ndarray        # [d_model] int16

    # FFN
    up_weight: np.ndarray       # [2*d_ff, d_model] int8 (gate+up combined)
    up_mult: np.int32           # scalar
    up_shift: np.int8           # scalar
    down_weight: np.ndarray     # [d_model, d_ff] int8
    down_mult: np.int32         # scalar
    down_shift: np.int8         # scalar

    # Post-FFN norm (Q1.14)
    post_ffn_norm: np.ndarray   # [d_model] int16


@dataclass
class ModelWeights:
    """Full model weights (czip-model-v1 integer-only format)."""
    config: ModelConfig
    embedding: np.ndarray       # [vocab_size, d_model] int8
    layers: List[LayerWeights]
    final_norm: np.ndarray      # [d_model] int16 (Q1.14)
    head_weight: np.ndarray     # [vocab_size, d_model] int8
    head_mult: np.int32         # scalar
    head_shift: np.int8         # scalar
    tokenizer_json: Optional[str] = None
    # Embedded LUTs for bit-exactness
    exp2_lut: Optional[np.ndarray] = None      # [256] uint16
    rope_cos_lut: Optional[np.ndarray] = None  # [max_seq_len, half_dim] int16
    rope_sin_lut: Optional[np.ndarray] = None  # [max_seq_len, half_dim] int16

    @classmethod
    def load(cls, path: Path | str, config: Optional[ModelConfig] = None) -> "ModelWeights":
        """Load model weights from czip-model-v1 safetensors file."""
        if not HAS_SAFETENSORS:
            raise RuntimeError("safetensors not installed. Install with: pip install safetensors")

        path = Path(path)

        with safe_open(path, framework="numpy") as f:
            metadata = f.metadata() or {}

            # Check format
            fmt = metadata.get("format", "")
            if fmt and fmt != '"czip-model-v1"' and fmt != "czip-model-v1":
                print(f"Warning: Unknown format '{fmt}', expected 'czip-model-v1'")

            # Load config from metadata if not provided
            if config is None:
                config = ModelConfig.from_metadata(metadata)

            def get_tensor(name: str) -> np.ndarray:
                if name not in f.keys():
                    raise KeyError(f"Missing tensor: {name}")
                return f.get_tensor(name)

            def get_scalar_i32(name: str, default: int = 1) -> np.int32:
                if name in f.keys():
                    return np.int32(f.get_tensor(name))
                return np.int32(default)

            def get_scalar_i8(name: str, default: int = 0) -> np.int8:
                if name in f.keys():
                    return np.int8(f.get_tensor(name))
                return np.int8(default)

            def ensure_i8(tensor: np.ndarray) -> np.ndarray:
                """Ensure tensor is int8."""
                if tensor.dtype != np.int8:
                    return tensor.astype(np.int8)
                return tensor

            def ensure_i16(tensor: np.ndarray) -> np.ndarray:
                """Ensure tensor is int16."""
                if tensor.dtype != np.int16:
                    return tensor.astype(np.int16)
                return tensor

            # Load embedding
            embedding = ensure_i8(get_tensor("embed.weight"))

            # Load layers
            layers = []
            for i in range(config.n_layers):
                layer = LayerWeights(
                    # Norms (int16 Q1.14)
                    attn_norm=ensure_i16(get_tensor(f"layers.{i}.attn_norm.weight")),
                    post_attn_norm=ensure_i16(get_tensor(f"layers.{i}.post_attn_norm.weight")),
                    ffn_norm=ensure_i16(get_tensor(f"layers.{i}.ffn_norm.weight")),
                    post_ffn_norm=ensure_i16(get_tensor(f"layers.{i}.post_ffn_norm.weight")),

                    # Attention
                    qkv_weight=ensure_i8(get_tensor(f"layers.{i}.attn.qkv.weight")),
                    qkv_mult=get_scalar_i32(f"layers.{i}.attn.qkv.mult"),
                    qkv_shift=get_scalar_i8(f"layers.{i}.attn.qkv.shift"),
                    out_weight=ensure_i8(get_tensor(f"layers.{i}.attn.out.weight")),
                    out_mult=get_scalar_i32(f"layers.{i}.attn.out.mult"),
                    out_shift=get_scalar_i8(f"layers.{i}.attn.out.shift"),
                    score_mult=ensure_i16(get_tensor(f"layers.{i}.attn.score_mult")),
                    score_shift=get_scalar_i8(f"layers.{i}.attn.score_shift", 15),

                    # FFN
                    up_weight=ensure_i8(get_tensor(f"layers.{i}.ffn.up.weight")),
                    up_mult=get_scalar_i32(f"layers.{i}.ffn.up.mult"),
                    up_shift=get_scalar_i8(f"layers.{i}.ffn.up.shift"),
                    down_weight=ensure_i8(get_tensor(f"layers.{i}.ffn.down.weight")),
                    down_mult=get_scalar_i32(f"layers.{i}.ffn.down.mult"),
                    down_shift=get_scalar_i8(f"layers.{i}.ffn.down.shift"),
                )
                layers.append(layer)

            # Load final norm and head
            final_norm = ensure_i16(get_tensor("norm.weight"))
            head_weight = ensure_i8(get_tensor("head.weight"))
            head_mult = get_scalar_i32("head.mult")
            head_shift = get_scalar_i8("head.shift")

            # Load tokenizer if present
            tokenizer_json = None
            if "tokenizer.data" in f.keys():
                tokenizer_data = f.get_tensor("tokenizer.data").tobytes()
                codec = metadata.get("tokenizer_codec", '"zstd"').strip('"')

                if codec == "zstd":
                    if not HAS_ZSTD:
                        print("Warning: zstd not installed, cannot decompress tokenizer")
                    else:
                        dctx = zstd.ZstdDecompressor()
                        tokenizer_json = dctx.decompress(tokenizer_data).decode('utf-8')
                elif codec == "brotli":
                    if not HAS_BROTLI:
                        print("Warning: brotli not installed, cannot decompress tokenizer")
                    else:
                        tokenizer_json = brotli.decompress(tokenizer_data).decode('utf-8')

            # Load embedded LUTs for bit-exactness
            exp2_lut = None
            rope_cos_lut = None
            rope_sin_lut = None
            if "lut.exp2" in f.keys():
                exp2_lut = f.get_tensor("lut.exp2").astype(np.uint16)
            if "lut.rope_cos" in f.keys():
                rope_cos_lut = f.get_tensor("lut.rope_cos").astype(np.int16)
            if "lut.rope_sin" in f.keys():
                rope_sin_lut = f.get_tensor("lut.rope_sin").astype(np.int16)

            return cls(
                config=config,
                embedding=embedding,
                layers=layers,
                final_norm=final_norm,
                head_weight=head_weight,
                head_mult=head_mult,
                head_shift=head_shift,
                tokenizer_json=tokenizer_json,
                exp2_lut=exp2_lut,
                rope_cos_lut=rope_cos_lut,
                rope_sin_lut=rope_sin_lut,
            )


def requantize_i32_to_i8(x: np.ndarray, mult: np.int32, shift: np.int8) -> np.ndarray:
    """Requantize int32 accumulator to int8 using shift-multiply.

    Uses arithmetic right shift with round-to-nearest (tie-to-even for consistency,
    but we use tie-to-away for simplicity which is common in quantized inference).

    output = clamp((x * mult + (1 << (shift-1))) >> shift, -127, 127)
    """
    if shift <= 0:
        # No shift, just multiply and clamp
        result = x.astype(np.int64) * int(mult)
        return np.clip(result, -127, 127).astype(np.int8)

    # Widen to int64 for intermediate computation
    x64 = x.astype(np.int64)
    mult64 = np.int64(mult)

    # Multiply
    product = x64 * mult64

    # Round-to-nearest with tie-to-away (add half, then shift)
    half = np.int64(1) << (int(shift) - 1)
    rounded = (product + half) >> int(shift)

    # Clamp to int8 range (symmetric: -127 to 127)
    return np.clip(rounded, -127, 127).astype(np.int8)


class Model:
    """Neural compression model (czip-model-v1 integer-only format)."""

    def __init__(self, weights: ModelWeights):
        """Create model from weights."""
        self.weights = weights
        self.config = weights.config

        # Create KV caches for each layer
        self.kv_caches = [
            KVCache.create(self.config.max_seq_len, self.config.head_dim)
            for _ in range(self.config.n_layers)
        ]

        # Use embedded LUTs for bit-exactness, fall back to generated if not present
        if weights.exp2_lut is not None:
            self.exp_lut = weights.exp2_lut
        else:
            self.exp_lut = get_exp2_lut()

        if weights.rope_cos_lut is not None and weights.rope_sin_lut is not None:
            self.rope_cos_lut = weights.rope_cos_lut
            self.rope_sin_lut = weights.rope_sin_lut
        else:
            cos_lut, sin_lut = get_rope_lut()
            self.rope_cos_lut = cos_lut
            self.rope_sin_lut = sin_lut

    def reset(self):
        """Reset KV caches for new sequence."""
        self.kv_caches = [
            KVCache.create(self.config.max_seq_len, self.config.head_dim)
            for _ in range(self.config.n_layers)
        ]

    def forward(self, token: int, pos: int) -> np.ndarray:
        """Forward pass for a single token, returns logits."""
        d = self.config.d_model
        d_ff = self.config.d_ff
        n_heads = self.config.n_heads
        head_dim = self.config.head_dim
        kv_dim = self.config.n_kv_heads * head_dim

        # Embedding lookup (int8)
        x = self.weights.embedding[token].copy()

        # Process each layer
        for layer_idx, layer in enumerate(self.weights.layers):
            # Pre-attention norm (int16 Q1.14 weights)
            normed = rmsnorm_i8_int(x, layer.attn_norm)

            # QKV projection (combined)
            # qkv_weight shape: [d_model + 2*kv_dim, d_model]
            qkv_i32 = linear_i8_to_i32(normed, layer.qkv_weight)

            # Split Q, K, V
            q_i32 = qkv_i32[:d]
            k_i32 = qkv_i32[d:d + kv_dim]
            v_i32 = qkv_i32[d + kv_dim:d + 2 * kv_dim]

            # Requantize to int8 using shift-multiply
            q = requantize_i32_to_i8(q_i32, layer.qkv_mult, layer.qkv_shift)
            k = requantize_i32_to_i8(k_i32, layer.qkv_mult, layer.qkv_shift)
            v = requantize_i32_to_i8(v_i32, layer.qkv_mult, layer.qkv_shift)

            # Post-QKV norm (int16 Q1.14 weights)
            q = rmsnorm_i8_int(q, layer.post_attn_norm)

            # Multi-head attention with MQA
            # First, update KV cache with current K/V (only once, before heads loop)
            kv_cache = self.kv_caches[layer_idx]

            # Apply RoPE to K and store to cache
            k_rot = apply_rope_i8(k, pos, self.rope_cos_lut, self.rope_sin_lut)
            kv_cache.k_cache[pos] = k_rot
            kv_cache.v_cache[pos] = v

            seq_len = pos + 1
            k_cache = kv_cache.k_cache[:seq_len]
            v_cache = kv_cache.v_cache[:seq_len]

            attn_outputs = []
            for head_idx in range(n_heads):
                head_start = head_idx * head_dim
                head_end = head_start + head_dim
                q_head = q[head_start:head_end]

                # Get per-head score multiplier
                score_mul = int(layer.score_mult[head_idx]) if hasattr(layer, 'score_mult') else 32768

                # Attention for this head (MQA: all heads share same K/V cache)
                head_out = gqa_attention_mqa_i8_cached(
                    q_head,
                    k_cache,
                    v_cache,
                    pos,
                    exp2_lut=self.exp_lut,
                    rope_cos=self.rope_cos_lut,
                    rope_sin=self.rope_sin_lut,
                    score_mul_q15=score_mul,
                )
                attn_outputs.append(head_out)

            attn_out = np.concatenate(attn_outputs)

            # Output projection
            o_i32 = linear_i8_to_i32(attn_out, layer.out_weight)
            o = requantize_i32_to_i8(o_i32, layer.out_mult, layer.out_shift)

            # Residual (int8 + int8 -> int8 with clamp)
            x = np.array([clamp_i8(int(x[i]) + int(o[i])) for i in range(d)], dtype=np.int8)

            # Pre-FFN norm (int16 Q1.14 weights)
            normed_ffn = rmsnorm_i8_int(x, layer.ffn_norm)

            # FFN: combined gate+up projection
            # up_weight shape: [2*d_ff, d_model]
            up_i32 = linear_i8_to_i32(normed_ffn, layer.up_weight)

            # Split gate and up
            gate_i32 = up_i32[:d_ff]
            up_i32_split = up_i32[d_ff:2 * d_ff]

            gate = requantize_i32_to_i8(gate_i32, layer.up_mult, layer.up_shift)
            up = requantize_i32_to_i8(up_i32_split, layer.up_mult, layer.up_shift)

            # ReGLU activation
            gated = reglu_i8(gate, up)

            # Down projection
            down_i32 = linear_i8_to_i32(gated, layer.down_weight)
            down = requantize_i32_to_i8(down_i32, layer.down_mult, layer.down_shift)

            # Post-FFN norm (applied after down projection, int16 Q1.14 weights)
            down = rmsnorm_i8_int(down, layer.post_ffn_norm)

            # Residual
            x = np.array([clamp_i8(int(x[i]) + int(down[i])) for i in range(d)], dtype=np.int8)

        # Final norm (int16 Q1.14 weights)
        final_normed = rmsnorm_i8_int(x, self.weights.final_norm)

        # LM head - returns int32 logits (not requantized for softmax)
        logits = linear_i8_to_i32(final_normed, self.weights.head_weight)

        return logits

    def get_cumfreqs(self, logits: np.ndarray) -> tuple:
        """Get cumulative frequencies from logits.

        Returns inclusive cumfreqs (no leading 0) matching CUDA.
        """
        return build_cumfreqs(logits, exp_lut=self.exp_lut)

    def get_tokenizer(self) -> Optional[dict]:
        """Get embedded tokenizer if present."""
        if self.weights.tokenizer_json:
            return json.loads(self.weights.tokenizer_json)
        return None
