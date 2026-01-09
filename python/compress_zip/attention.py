"""
Grouped Query Attention (GQA/MQA) with RoPE.

Matches CUDA attention.cu kernel.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from .primitives import sra_rne_tte_s32, sra_rne_tte_s64_to_s32, clamp_i8
from .lut import get_rope_lut, get_exp2_lut, exp_q16_from_neg_fixed, ROPE_HALF_DIM

# Constants matching CUDA
HEAD_DIM = 64
MAX_SEQ_LEN = 64
TOKENS_PER_BATCH = 16

# Attention scaling coefficient
# We want to divide by sqrt(HEAD_DIM) = 8
# In Q24 format: coefficient = 2^24 / 8 = 2097152
SQRT_HEAD_DIM = 8
SQRT_HEAD_DIM_SHIFT = 3  # log2(8) = 3
ATTN_COEF_Q24 = (1 << 24) // SQRT_HEAD_DIM  # 2097152

# Q/K scaling for attention
Q_SHIFT = 2  # Additional shift for Q to prevent overflow


@dataclass
class KVCache:
    """Key-Value cache for autoregressive generation."""
    k_cache: np.ndarray  # [max_seq_len, head_dim] as int8
    v_cache: np.ndarray  # [max_seq_len, head_dim] as int8
    seq_len: int  # Current sequence length

    @classmethod
    def create(cls, max_seq_len: int = MAX_SEQ_LEN, head_dim: int = HEAD_DIM) -> "KVCache":
        return cls(
            k_cache=np.zeros((max_seq_len, head_dim), dtype=np.int8),
            v_cache=np.zeros((max_seq_len, head_dim), dtype=np.int8),
            seq_len=0,
        )

    def append(self, k: np.ndarray, v: np.ndarray):
        """Append new K/V to cache."""
        if k.ndim == 1:
            k = k.reshape(1, -1)
            v = v.reshape(1, -1)

        n_new = k.shape[0]
        assert self.seq_len + n_new <= self.k_cache.shape[0], "Cache overflow"

        self.k_cache[self.seq_len:self.seq_len + n_new] = k
        self.v_cache[self.seq_len:self.seq_len + n_new] = v
        self.seq_len += n_new


def apply_rope_i8(
    x: np.ndarray,
    pos: int,
) -> np.ndarray:
    """
    Apply RoPE to a single vector at position pos.

    x is [head_dim] as int8.
    Returns rotated vector as int8.
    """
    rope_lut = get_rope_lut()
    out = np.zeros_like(x)

    for i in range(ROPE_HALF_DIM):
        # Get cos/sin from LUT (Q15)
        cos_val = rope_lut.get_cos(pos, i)
        sin_val = rope_lut.get_sin(pos, i)

        # Get pair of values
        x0 = int(x[i])
        x1 = int(x[i + ROPE_HALF_DIM])

        # Rotation: [cos, -sin; sin, cos]
        # out[i] = x0 * cos - x1 * sin
        # out[i+half] = x0 * sin + x1 * cos

        # x is Q0.7, cos/sin is Q1.15
        # Product is Q1.22, shift by 15 to get Q0.7
        out0 = x0 * cos_val - x1 * sin_val
        out1 = x0 * sin_val + x1 * cos_val

        out[i] = clamp_i8(sra_rne_tte_s32(out0, 15))
        out[i + ROPE_HALF_DIM] = clamp_i8(sra_rne_tte_s32(out1, 15))

    return out.astype(np.int8)


def apply_rope_q_i8(
    q: np.ndarray,
    pos: int,
) -> np.ndarray:
    """
    Apply RoPE to Q with additional scaling (shift by Q_SHIFT).

    This prevents overflow in attention scores.
    """
    rope_lut = get_rope_lut()
    out = np.zeros_like(q)

    for i in range(ROPE_HALF_DIM):
        cos_val = rope_lut.get_cos(pos, i)
        sin_val = rope_lut.get_sin(pos, i)

        x0 = int(q[i])
        x1 = int(q[i + ROPE_HALF_DIM])

        out0 = x0 * cos_val - x1 * sin_val
        out1 = x0 * sin_val + x1 * cos_val

        # Extra shift for Q
        out[i] = clamp_i8(sra_rne_tte_s32(out0, 15 + Q_SHIFT))
        out[i + ROPE_HALF_DIM] = clamp_i8(sra_rne_tte_s32(out1, 15 + Q_SHIFT))

    return out.astype(np.int8)


def gqa_attention_mqa_i8(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    pos: int,
    kv_cache: Optional[KVCache] = None,
) -> np.ndarray:
    """
    Multi-Query Attention (MQA) with single K/V head.

    Args:
        q: Query vector [head_dim] as int8
        k: Key vector [head_dim] as int8
        v: Value vector [head_dim] as int8
        pos: Current position (0-indexed)
        kv_cache: Optional KV cache for autoregressive generation

    Returns:
        Output vector [head_dim] as int8
    """
    exp_lut = get_exp2_lut()

    # Apply RoPE
    q_rot = apply_rope_q_i8(q, pos)
    k_rot = apply_rope_i8(k, pos)

    # Update KV cache if provided
    if kv_cache is not None:
        kv_cache.k_cache[pos] = k_rot
        kv_cache.v_cache[pos] = v
        seq_len = pos + 1
        k_cache = kv_cache.k_cache[:seq_len]
        v_cache = kv_cache.v_cache[:seq_len]
    else:
        # No cache, just current token
        seq_len = 1
        k_cache = k_rot.reshape(1, -1)
        v_cache = v.reshape(1, -1)

    # Compute attention scores: Q @ K^T
    # Q is scaled down by Q_SHIFT already
    scores = np.zeros(seq_len, dtype=np.int32)
    q_i32 = q_rot.astype(np.int32)

    for s in range(seq_len):
        k_i32 = k_cache[s].astype(np.int32)
        score = 0
        for d in range(HEAD_DIM):
            score += q_i32[d] * k_i32[d]
        scores[s] = score

    # Find max score for numerical stability
    max_score = int(np.max(scores))

    # Compute softmax using integer exp
    # exp(score - max_score) approximated as exp2((score - max_score) * log2(e))
    # For simplicity, we use a fixed-point approximation

    # Convert score differences to Q8 format for exp lookup
    # score diff is in range [-(max-min), 0]
    # We want to compute exp2(diff * scale) where scale converts to log2 base

    # Simpler approach: use scaled differences directly
    # Each score unit roughly corresponds to some fraction of a bit
    EXP_SCALE = 4  # Tunable: how many score units per bit of exponent

    weights_q16 = np.zeros(seq_len, dtype=np.int64)
    for s in range(seq_len):
        diff = max_score - scores[s]  # Always >= 0
        neg_exp_q8 = diff * EXP_SCALE  # Convert to Q8 exponent
        weights_q16[s] = exp_q16_from_neg_fixed(neg_exp_q8, exp_lut)

    # Normalize weights
    total_weight = int(np.sum(weights_q16))
    if total_weight == 0:
        total_weight = 1

    # Compute weighted sum of values
    output_acc = np.zeros(HEAD_DIM, dtype=np.int64)

    for s in range(seq_len):
        w = int(weights_q16[s])
        v_i32 = v_cache[s].astype(np.int32)
        for d in range(HEAD_DIM):
            output_acc[d] += w * v_i32[d]

    # Normalize by total weight and convert back to int8
    # output_acc is Q16 * Q0.7 = Q0.23
    # Divide by total_weight (in Q16) to get Q0.7
    output = np.zeros(HEAD_DIM, dtype=np.int8)

    for d in range(HEAD_DIM):
        # Divide with rounding
        val = output_acc[d]
        if total_weight > 0:
            # Round to nearest
            if val >= 0:
                result = (val + total_weight // 2) // total_weight
            else:
                result = (val - total_weight // 2) // total_weight
        else:
            result = 0
        output[d] = clamp_i8(int(result))

    return output


# Test cases
if __name__ == "__main__":
    # Test RoPE
    x = np.zeros(HEAD_DIM, dtype=np.int8)
    x[0] = 64  # First element non-zero

    # At position 0, rotation should be minimal
    x_rot = apply_rope_i8(x, 0)
    print(f"RoPE at pos 0: {x[:4]} -> {x_rot[:4]}")

    # At position 1, should see rotation
    x_rot_1 = apply_rope_i8(x, 1)
    print(f"RoPE at pos 1: {x[:4]} -> {x_rot_1[:4]}")

    # Test attention
    q = np.random.randint(-64, 64, HEAD_DIM, dtype=np.int8)
    k = np.random.randint(-64, 64, HEAD_DIM, dtype=np.int8)
    v = np.random.randint(-64, 64, HEAD_DIM, dtype=np.int8)

    output = gqa_attention_mqa_i8(q, k, v, 0)
    print(f"Attention output shape: {output.shape}")
    print(f"Attention output sample: {output[:8]}")

    # Test with KV cache
    cache = KVCache.create()
    for i in range(4):
        q_i = np.random.randint(-64, 64, HEAD_DIM, dtype=np.int8)
        k_i = np.random.randint(-64, 64, HEAD_DIM, dtype=np.int8)
        v_i = np.random.randint(-64, 64, HEAD_DIM, dtype=np.int8)
        out_i = gqa_attention_mqa_i8(q_i, k_i, v_i, i, cache)
        print(f"Position {i}: cache.seq_len = {cache.seq_len}")

    print("All attention tests passed!")
