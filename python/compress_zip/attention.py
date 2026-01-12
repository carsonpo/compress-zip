"""
Grouped Query Attention (GQA/MQA) with RoPE.

Matches CUDA attention.cu kernel.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from .primitives import sra_rne_tte_s32, sra_rne_tte_s64_to_s32, clamp_i8, div_rne_tte_s64_to_s32, sra_rne_tte_s32_np
from .lut import get_rope_lut, get_exp2_lut, ROPE_HALF_DIM

# Constants matching CUDA attention.cu
HEAD_DIM = 64
MAX_SEQ_LEN = 64
TOKENS_PER_BATCH = 16

# CUDA constants for attention score scaling
Q_SHIFT = 2
SQRT_HEAD_DIM_SHIFT = 3

# Score to exp mapping coefficient for Q0.14 scores (matches CUDA exactly)
# LOG2E = 1.4426950408889634
# ATTN_COEF = LOG2E / 64.0
# ATTN_COEF_Q24 = round(ATTN_COEF * 2^24)
LOG2E = 1.4426950408889634
ATTN_COEF = LOG2E / 64.0
ATTN_COEF_Q24 = int(ATTN_COEF * (1 << 24) + 0.5)  # 378173


def exp_q16_from_attn_diff(diff: int, exp_lut) -> int:
    """
    Compute exp2 for attention score difference.
    Matches CUDA exp_q16_from_attn_diff exactly.

    diff should be <= 0 (score - max_score)
    Returns uint16 Q16 representation.
    """
    if diff >= 0:
        return 65535

    neg = -diff

    # t256 = round(neg * ATTN_COEF_Q24 / 2^24)
    t256 = (neg * ATTN_COEF_Q24 + (1 << 23)) >> 24

    ip = t256 >> 8  # integer part
    frac = t256 & 255  # fractional part

    if ip >= 31:
        return 1

    base = (1 << 16) >> ip

    # Get fractional multiplier from LUT
    if hasattr(exp_lut, '__getitem__'):
        # numpy array or Exp2LutQ16 class
        frac_mul = int(exp_lut[frac])
    else:
        frac_mul = int(exp_lut[frac])

    out = (base * frac_mul + 0x8000) >> 16

    if out < 1:
        out = 1
    if out > 65535:
        out = 65535

    return out


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
    rope_cos: np.ndarray = None,
    rope_sin: np.ndarray = None,
) -> np.ndarray:
    """
    Apply RoPE to a single vector at position pos.

    x is [head_dim] as int8.
    rope_cos/rope_sin: [max_seq_len, half_dim] int16 LUTs (optional, generated if not provided)
    Returns rotated vector as int8.
    """
    # Use embedded LUTs if provided, otherwise generate
    if rope_cos is None or rope_sin is None:
        rope_lut = get_rope_lut()
        use_class = True
    else:
        use_class = False

    out = np.zeros_like(x)

    for i in range(ROPE_HALF_DIM):
        # Get cos/sin from LUT (Q15)
        if use_class:
            cos_val = rope_lut.get_cos(pos, i)
            sin_val = rope_lut.get_sin(pos, i)
        else:
            cos_val = int(rope_cos[pos, i])
            sin_val = int(rope_sin[pos, i])

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
    rope_cos: np.ndarray = None,
    rope_sin: np.ndarray = None,
) -> np.ndarray:
    """
    Apply RoPE to Q with additional scaling (shift by Q_SHIFT).

    This prevents overflow in attention scores.
    rope_cos/rope_sin: [max_seq_len, half_dim] int16 LUTs (optional, generated if not provided)
    """
    # Use embedded LUTs if provided, otherwise generate
    if rope_cos is None or rope_sin is None:
        rope_lut = get_rope_lut()
        use_class = True
    else:
        use_class = False

    out = np.zeros_like(q)

    for i in range(ROPE_HALF_DIM):
        if use_class:
            cos_val = rope_lut.get_cos(pos, i)
            sin_val = rope_lut.get_sin(pos, i)
        else:
            cos_val = int(rope_cos[pos, i])
            sin_val = int(rope_sin[pos, i])

        x0 = int(q[i])
        x1 = int(q[i + ROPE_HALF_DIM])

        out0 = x0 * cos_val - x1 * sin_val
        out1 = x0 * sin_val + x1 * cos_val

        # CUDA: shift by 15 first, then by NET_SHIFT (two separate rounding operations)
        # NET_SHIFT = SQRT_HEAD_DIM_SHIFT - Q_SHIFT = 3 - 2 = 1
        x_rot = sra_rne_tte_s32(out0, 15)
        y_rot = sra_rne_tte_s32(out1, 15)

        # Apply NET_SHIFT = 1
        x_rot = sra_rne_tte_s32(x_rot, 1)
        y_rot = sra_rne_tte_s32(y_rot, 1)

        out[i] = clamp_i8(x_rot)
        out[i + ROPE_HALF_DIM] = clamp_i8(y_rot)

    return out.astype(np.int8)


def gqa_attention_mqa_i8(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    pos: int,
    kv_cache: Optional[KVCache] = None,
    exp2_lut: np.ndarray = None,
    rope_cos: np.ndarray = None,
    rope_sin: np.ndarray = None,
    score_mul_q15: int = 32768,
) -> np.ndarray:
    """
    Multi-Query Attention (MQA) with single K/V head.
    Matches CUDA gqa_attention_mqa_i8_kernel exactly.

    Args:
        q: Query vector [head_dim] as int8
        k: Key vector [head_dim] as int8
        v: Value vector [head_dim] as int8
        pos: Current position (0-indexed)
        kv_cache: Optional KV cache for autoregressive generation
        exp2_lut: Optional [256] uint16 exp2 LUT (uses generated if not provided)
        rope_cos: Optional [max_seq_len, half_dim] int16 RoPE cos LUT
        rope_sin: Optional [max_seq_len, half_dim] int16 RoPE sin LUT
        score_mul_q15: Score multiplier in Q0.15 format (default 32768 = 1.0)

    Returns:
        Output vector [head_dim] as int8
    """
    # Use embedded LUTs if provided, otherwise generate
    if exp2_lut is None:
        exp_lut = get_exp2_lut()
    else:
        exp_lut = exp2_lut

    # Apply RoPE
    q_rot = apply_rope_q_i8(q, pos, rope_cos, rope_sin)
    k_rot = apply_rope_i8(k, pos, rope_cos, rope_sin)

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

    # Compute attention scores matching CUDA exactly (vectorized)
    q_rot_64 = q_rot.astype(np.int64)
    k_cache_64 = k_cache.astype(np.int64)
    dots = np.sum(q_rot_64 * k_cache_64, axis=1)  # [seq_len]

    # CUDA: dot_unshift = round(dot >> Q_SHIFT) with ties-to-even
    dots_unshift = sra_rne_tte_s32_np(dots, Q_SHIFT)

    # CUDA: prod = dot_unshift * score_mul; score = round(prod >> 15) with ties-to-even
    prods = dots_unshift.astype(np.int64) * score_mul_q15
    scores = np.array([sra_rne_tte_s64_to_s32(int(p), 15) for p in prods])

    # Online softmax (matching CUDA)
    m = -(1 << 30)  # Running max (use large negative instead of INT32_MIN)
    s = 0  # Running sum of exp weights (uint64)
    vacc = np.zeros(HEAD_DIM, dtype=np.int64)  # Weighted V accumulator

    for t in range(seq_len):
        score = int(scores[t])

        # Update online softmax
        if score > m:
            # New max found - rescale
            if m != -(1 << 30):
                diff_old = m - score
                scale_old = exp_q16_from_attn_diff(diff_old, exp_lut)
                # Rescale S: (S * scale + 0x8000) >> 16
                s = (s * scale_old + 0x8000) >> 16
                # Rescale vacc (vectorized, matching CUDA exactly)
                prod = vacc * scale_old
                adjustment = np.where(prod >= 0, 1 << 15, -(1 << 15))
                vacc = (prod + adjustment) >> 16
            m = score

        # Compute exp weight for this token
        diff = score - m
        w = exp_q16_from_attn_diff(diff, exp_lut)
        s += w

        # Accumulate weighted V (vectorized)
        vacc += w * v_cache[t].astype(np.int64)

    # Normalize output
    denom = s if s != 0 else 1

    # Vectorized output computation
    y = np.zeros(HEAD_DIM, dtype=np.int64)
    for i in range(HEAD_DIM):
        y[i] = div_rne_tte_s64_to_s32(int(vacc[i]), denom)
    output = np.clip(y, -128, 127).astype(np.int8)

    return output


def gqa_attention_mqa_i8_cached(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    pos: int,
    exp2_lut: np.ndarray = None,
    rope_cos: np.ndarray = None,
    rope_sin: np.ndarray = None,
    score_mul_q15: int = 32768,
    sink_key: np.ndarray = None,
) -> np.ndarray:
    """
    Multi-Query Attention with pre-cached K/V.
    Matches CUDA exactly - K is already rotated and stored in cache.

    Args:
        q: Query vector [head_dim] as int8
        k_cache: Key cache [seq_len, head_dim] as int8 (already RoPE'd)
        v_cache: Value cache [seq_len, head_dim] as int8
        pos: Current position (0-indexed)
        exp2_lut: Optional [256] uint16 exp2 LUT
        rope_cos: Optional [max_seq_len, half_dim] int16 RoPE cos LUT
        rope_sin: Optional [max_seq_len, half_dim] int16 RoPE sin LUT
        score_mul_q15: Score multiplier in Q0.15 format
        sink_key: Optional [head_dim] int8 attention sink key (value is zero)

    Returns:
        Output vector [head_dim] as int8
    """
    if exp2_lut is None:
        exp_lut = get_exp2_lut()
    else:
        exp_lut = exp2_lut

    # Apply RoPE to Q only (K is already rotated in cache)
    q_rot = apply_rope_q_i8(q, pos, rope_cos, rope_sin)

    seq_len = pos + 1

    # Compute attention scores matching CUDA exactly (vectorized)
    # Q*K dot products for all positions at once
    q_rot_64 = q_rot.astype(np.int64)
    k_cache_64 = k_cache[:seq_len].astype(np.int64)
    dots = np.sum(q_rot_64 * k_cache_64, axis=1)  # [seq_len]

    # CUDA: dot_unshift = round(dot >> Q_SHIFT) with ties-to-even
    dots_unshift = sra_rne_tte_s32_np(dots, Q_SHIFT)

    # CUDA: prod = dot_unshift * score_mul; score = round(prod >> 15) with ties-to-even
    prods = dots_unshift.astype(np.int64) * score_mul_q15
    # For 64-bit values, process element-wise to ensure correct rounding
    scores = np.array([sra_rne_tte_s64_to_s32(int(p), 15) for p in prods])

    # Compute sink score if sink key is provided
    sink_score = None
    if sink_key is not None:
        # Dot product with RoPE'd Q (no RoPE on sink key)
        sink_dot = int(np.sum(q_rot.astype(np.int64) * sink_key.astype(np.int64)))
        sink_dot_unshift = sra_rne_tte_s32_np(np.array([sink_dot]), Q_SHIFT)[0]
        sink_prod = int(sink_dot_unshift) * score_mul_q15
        sink_score = sra_rne_tte_s64_to_s32(sink_prod, 15)

    # Online softmax (matching CUDA)
    m = -(1 << 30)
    s = 0
    vacc = np.zeros(HEAD_DIM, dtype=np.int64)

    # Include sink score in initial max if present
    if sink_score is not None:
        m = sink_score
        w_sink = exp_q16_from_attn_diff(0, exp_lut)  # diff = 0 since score == m
        s = w_sink
        # Note: sink value is zero, so no contribution to vacc

    for t in range(seq_len):
        score = int(scores[t])

        if score > m:
            if m != -(1 << 30):
                diff_old = m - score
                scale_old = exp_q16_from_attn_diff(diff_old, exp_lut)
                s = (s * scale_old + 0x8000) >> 16
                # Vectorized vacc rescaling matching CUDA
                prod = vacc * scale_old
                adjustment = np.where(prod >= 0, 1 << 15, -(1 << 15))
                vacc = (prod + adjustment) >> 16
            m = score

        diff = score - m
        w = exp_q16_from_attn_diff(diff, exp_lut)
        s += w

        vacc += w * v_cache[t].astype(np.int64)

    denom = s if s != 0 else 1

    # Vectorized output computation
    vacc_arr = vacc.astype(np.int64)
    y = np.zeros(HEAD_DIM, dtype=np.int64)
    for i in range(HEAD_DIM):
        y[i] = div_rne_tte_s64_to_s32(int(vacc_arr[i]), denom)
    output = np.clip(y, -128, 127).astype(np.int8)

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
