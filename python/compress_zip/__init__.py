"""Deterministic CPU implementation of neural compression."""

from .primitives import (
    sra_rne_tte_s32,
    sra_rne_tte_s64_to_s32,
    udiv_rne_tte_u32,
    div_rne_tte_s32,
    isqrt32_restoring,
    clamp_i8,
    argmax_deterministic,
)
from .lut import Exp2LutQ16, RopeLut, exp_q16_from_neg_fixed
from .rmsnorm import rmsnorm_i8, compute_inv_rms_q15_from_mean_sq
from .reglu import reglu_i8
from .linear import linear_i8_to_i32
from .attention import gqa_attention_mqa_i8, KVCache
from .softmax_cdf import build_cumfreqs
from .arith_coder import ArithEncoder, ArithDecoder
from .file_format import CZIPv1OuterHeader, CZIPv1InnerHeader, CompressedFile, Codec

__all__ = [
    # primitives
    "sra_rne_tte_s32",
    "sra_rne_tte_s64_to_s32",
    "udiv_rne_tte_u32",
    "div_rne_tte_s32",
    "isqrt32_restoring",
    "clamp_i8",
    "argmax_deterministic",
    # lut
    "Exp2LutQ16",
    "RopeLut",
    "exp_q16_from_neg_fixed",
    # rmsnorm
    "rmsnorm_i8",
    "compute_inv_rms_q15_from_mean_sq",
    # reglu
    "reglu_i8",
    # linear
    "linear_i8_to_i32",
    # attention
    "gqa_attention_mqa_i8",
    "KVCache",
    # softmax_cdf
    "build_cumfreqs",
    # arith_coder
    "ArithEncoder",
    "ArithDecoder",
    # file_format
    "CZIPv1OuterHeader",
    "CZIPv1InnerHeader",
    "CompressedFile",
    "Codec",
]
