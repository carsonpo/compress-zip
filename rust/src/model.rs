//! Neural compression model implementation.
//!
//! Small transformer model for text compression using int8 arithmetic.
//! Supports czip-model-v1 format (integer-only: pre-quantized int8 weights with
//! shift-multiply requantization and int16 Q1.14 norm weights).

use crate::attention::{gqa_attention_mqa_i8, KVCache, HEAD_DIM, MAX_SEQ_LEN};
use crate::linear::linear_i8_to_i32;
use crate::lut::{Exp2LutQ16, RopeLut};
use crate::primitives::clamp_i8;
use crate::reglu::reglu_i8;
use crate::rmsnorm::{rmsnorm_i8_to_i8, compute_eps_scaled};
use crate::safetensors::{load_safetensors_with_metadata, Tensor};
use crate::softmax_cdf::{build_cumfreqs, compute_coef_q24_for_acc, compute_target_total};

use std::path::Path;

/// Model configuration
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub context_length: usize,  // Tokens per chunk (RoPE LUT size)
    pub rope_theta_q16: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1024,
            n_layers: 1,
            n_heads: 6,
            n_kv_heads: 1,
            d_model: 384,
            d_ff: 1024,
            head_dim: HEAD_DIM,
            max_seq_len: MAX_SEQ_LEN,
            context_length: MAX_SEQ_LEN,
            rope_theta_q16: 655360000, // 10000.0 * 65536
        }
    }
}

/// Transformer layer weights (czip-model-v1 integer-only format)
#[derive(Clone)]
pub struct LayerWeights {
    // Pre-attention norm (Q0.7 int8 - matches CUDA)
    pub attn_norm: Vec<i8>,         // [d_model]

    // Attention
    pub qkv_weight: Vec<i8>,        // [d_model + 2*kv_dim, d_model]
    pub qkv_mult: i32,
    pub qkv_shift: i8,
    pub out_weight: Vec<i8>,        // [d_model, d_model]
    pub out_mult: i32,
    pub out_shift: i8,
    pub score_mult: Vec<i32>,       // [n_heads] Q0.15 (stored as i32 since values can be > 32768)
    pub score_shift: i8,

    // Post-attention norm / Pre-FFN norm (Q0.7 int8)
    pub post_attn_norm: Vec<i8>,    // [d_model]

    // FFN norm / Post-attention norm in CUDA (Q0.7 int8)
    pub ffn_norm: Vec<i8>,          // [d_model]

    // FFN
    pub up_weight: Vec<i8>,         // [2*d_ff, d_model] (gate+up combined)
    pub up_mult: i32,
    pub up_shift: i8,
    pub down_weight: Vec<i8>,       // [d_model, d_ff]
    pub down_mult: i32,
    pub down_shift: i8,

    // Post-FFN norm (Q0.7 int8)
    pub post_ffn_norm: Vec<i8>,     // [d_model]
}

/// Full model weights (czip-model-v1 integer-only format)
#[derive(Clone)]
pub struct ModelWeights {
    pub config: ModelConfig,
    pub embedding: Vec<i8>,         // [vocab_size, d_model]
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<i8>,        // [d_model] Q0.7 int8
    pub head_weight: Vec<i8>,       // [vocab_size, d_model]
    pub head_mult: i32,
    pub head_shift: i8,
    // Embedded data from czip-model-v1 format
    pub tokenizer_data: Option<Vec<u8>>,   // Compressed tokenizer JSON
    pub tokenizer_codec: String,            // "zstd" or "brotli"
    pub exp2_lut: Vec<u16>,                // [256] uint16
    pub rope_cos_lut: Vec<i16>,            // [max_seq_len, half_dim] int16
    pub rope_sin_lut: Vec<i16>,            // [max_seq_len, half_dim] int16
    // Softmax CDF constants for bit-exactness
    pub coef_q24: Option<u32>,             // Coefficient for exp scaling
    pub target_total: Option<u32>,         // Target total for frequency table
}

impl ModelWeights {
    /// Load model weights from czip-model-v1 safetensors file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let (tensors, metadata) = load_safetensors_with_metadata(path)
            .map_err(|e| format!("Failed to load safetensors: {}", e))?;

        // Parse config from metadata
        let parse_usize = |key: &str, default: usize| -> usize {
            metadata.get(key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let parse_u32 = |key: &str, default: u32| -> u32 {
            metadata.get(key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };

        // Parse context_length (tokens per chunk / RoPE LUT size)
        // Prefer context_length, fall back to context_length_log2, then max_seq_len
        let context_length = if let Some(cl) = metadata.get("context_length").and_then(|v| v.parse::<usize>().ok()) {
            cl
        } else if let Some(cl_log2) = metadata.get("context_length_log2").and_then(|v| v.parse::<usize>().ok()) {
            1usize << cl_log2
        } else {
            parse_usize("max_seq_len", MAX_SEQ_LEN)
        };

        let mut config = ModelConfig {
            vocab_size: parse_usize("vocab_size", 1024),
            n_layers: parse_usize("n_layers", 1),
            n_heads: parse_usize("n_heads", 6),
            n_kv_heads: parse_usize("n_kv_heads", 1),
            d_model: parse_usize("d_model", 384),
            d_ff: parse_usize("d_ff", 1024),
            head_dim: parse_usize("head_dim", HEAD_DIM),
            max_seq_len: parse_usize("max_seq_len", MAX_SEQ_LEN),
            context_length,
            rope_theta_q16: parse_u32("rope_theta_q16", 655360000),
        };

        let tokenizer_codec = metadata.get("tokenizer_codec")
            .cloned()
            .unwrap_or_else(|| "zstd".to_string());

        // Helper to get tensor
        let get_tensor = |name: &str| -> Result<&Tensor, String> {
            tensors.get(name).ok_or_else(|| format!("Missing tensor: {}", name))
        };

        // Helper to get scalar i32
        let get_scalar_i32 = |name: &str| -> i32 {
            tensors.get(name)
                .map(|t| t.as_i32()[0])
                .unwrap_or(1)
        };

        // Helper to get scalar i8
        let get_scalar_i8 = |name: &str, default: i8| -> i8 {
            tensors.get(name)
                .map(|t| t.as_i8()[0])
                .unwrap_or(default)
        };

        // Convert tensor to i8 vec
        let to_i8_vec = |t: &Tensor| -> Vec<i8> {
            t.as_i8().to_vec()
        };

        // Convert tensor to i16 vec
        let to_i16_vec = |t: &Tensor| -> Vec<i16> {
            t.as_i16().to_vec()
        };

        // Convert tensor to u16 vec
        let to_u16_vec = |t: &Tensor| -> Vec<u16> {
            t.as_u16().to_vec()
        };

        // Convert tensor to i32 vec
        let to_i32_vec = |t: &Tensor| -> Vec<i32> {
            t.as_i32().to_vec()
        };

        // Load embedding
        let embedding = to_i8_vec(get_tensor("embed.weight")?);

        // Infer config from embedding size if not specified
        let embed_size = embedding.len();
        if config.d_model == 0 {
            config.d_model = embed_size / config.vocab_size;
        }

        // Load layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("layers.{}", i);

            let layer = LayerWeights {
                // Norms (int8 Q0.7 - matches CUDA)
                attn_norm: to_i8_vec(get_tensor(&format!("{}.attn_norm.weight", prefix))?),
                post_attn_norm: to_i8_vec(get_tensor(&format!("{}.post_attn_norm.weight", prefix))?),
                ffn_norm: to_i8_vec(get_tensor(&format!("{}.ffn_norm.weight", prefix))?),
                post_ffn_norm: to_i8_vec(get_tensor(&format!("{}.post_ffn_norm.weight", prefix))?),

                // Attention
                qkv_weight: to_i8_vec(get_tensor(&format!("{}.attn.qkv.weight", prefix))?),
                qkv_mult: get_scalar_i32(&format!("{}.attn.qkv.mult", prefix)),
                qkv_shift: get_scalar_i8(&format!("{}.attn.qkv.shift", prefix), 0),
                out_weight: to_i8_vec(get_tensor(&format!("{}.attn.out.weight", prefix))?),
                out_mult: get_scalar_i32(&format!("{}.attn.out.mult", prefix)),
                out_shift: get_scalar_i8(&format!("{}.attn.out.shift", prefix), 0),
                score_mult: to_i32_vec(get_tensor(&format!("{}.attn.score_mult", prefix))?),
                score_shift: get_scalar_i8(&format!("{}.attn.score_shift", prefix), 15),

                // FFN
                up_weight: to_i8_vec(get_tensor(&format!("{}.ffn.up.weight", prefix))?),
                up_mult: get_scalar_i32(&format!("{}.ffn.up.mult", prefix)),
                up_shift: get_scalar_i8(&format!("{}.ffn.up.shift", prefix), 0),
                down_weight: to_i8_vec(get_tensor(&format!("{}.ffn.down.weight", prefix))?),
                down_mult: get_scalar_i32(&format!("{}.ffn.down.mult", prefix)),
                down_shift: get_scalar_i8(&format!("{}.ffn.down.shift", prefix), 0),
            };
            layers.push(layer);
        }

        // Load final norm and head
        let final_norm = to_i8_vec(get_tensor("norm.weight")?);
        let head_weight = to_i8_vec(get_tensor("head.weight")?);
        let head_mult = get_scalar_i32("head.mult");
        let head_shift = get_scalar_i8("head.shift", 0);

        // Load embedded tokenizer data (optional)
        let tokenizer_data = tensors.get("tokenizer.data")
            .map(|t| t.as_u8().to_vec());

        // Load LUTs
        let exp2_lut = to_u16_vec(get_tensor("lut.exp2")?);
        let rope_cos_lut = to_i16_vec(get_tensor("lut.rope_cos")?);
        let rope_sin_lut = to_i16_vec(get_tensor("lut.rope_sin")?);

        // Load softmax CDF constants (optional, fall back to computed values)
        let coef_q24 = tensors.get("softmax.coef_q24")
            .map(|t| t.as_u32()[0]);
        let target_total = tensors.get("softmax.target_total")
            .map(|t| t.as_u32()[0]);

        Ok(Self {
            config,
            embedding,
            layers,
            final_norm,
            head_weight,
            head_mult,
            head_shift,
            tokenizer_data,
            tokenizer_codec,
            exp2_lut,
            rope_cos_lut,
            rope_sin_lut,
            coef_q24,
            target_total,
        })
    }
}

/// Requantize int32 accumulator to int8 using shift-multiply with ties-to-even.
///
/// Uses the same rounding as CUDA's sra_round_ties_to_even_i64.
fn requantize_i32_to_i8(x: &[i32], mult: i32, shift: i8) -> Vec<i8> {
    use crate::primitives::sra_rne_tte_s64_to_s32;

    if shift <= 0 {
        return x.iter()
            .map(|&v| {
                let result = (v as i64) * (mult as i64);
                result.clamp(-127, 127) as i8
            })
            .collect();
    }

    x.iter()
        .map(|&v| {
            let product = (v as i64) * (mult as i64);
            // Use ties-to-even rounding to match CUDA exactly
            let rounded = sra_rne_tte_s64_to_s32(product, shift as u32);
            rounded.clamp(-127, 127) as i8
        })
        .collect()
}

/// Neural compression model (czip-model-v1 integer-only format)
pub struct Model {
    pub weights: ModelWeights,
    pub kv_caches: Vec<KVCache>,
    pub rope_lut: RopeLut,
    pub exp_lut: Exp2LutQ16,
    pub coef_q24: u32,
    pub target_total: u32,
    pub eps_scaled: i32,  // Pre-computed epsilon for rmsnorm
}

impl Model {
    /// Create model from weights
    pub fn new(weights: ModelWeights) -> Self {
        let config = &weights.config;

        // Create KV caches for each layer
        let kv_caches: Vec<_> = (0..config.n_layers)
            .map(|_| KVCache::new(1, config.max_seq_len, config.head_dim))
            .collect();

        // Load LUTs from embedded data in weights (for bit-exactness)
        let exp_lut = Exp2LutQ16::from_array(&weights.exp2_lut)
            .expect("Invalid exp2 LUT in model weights");
        // Use context_length from config for RoPE LUT (tokens per chunk)
        let half_dim = config.head_dim / 2;
        let rope_lut = RopeLut::from_arrays_with_dims(
            &weights.rope_cos_lut,
            &weights.rope_sin_lut,
            config.context_length,
            half_dim,
        ).expect("Invalid RoPE LUT in model weights");

        // Use embedded constants for bit-exactness, fall back to computed if not present
        let coef_q24 = weights.coef_q24.unwrap_or_else(|| compute_coef_q24_for_acc(128.0));
        let target_total = weights.target_total.unwrap_or_else(|| compute_target_total(config.vocab_size));

        // Compute eps_scaled for rmsnorm (1e-5 is standard)
        let eps_scaled = compute_eps_scaled(1e-5);

        Self {
            weights,
            kv_caches,
            rope_lut,
            exp_lut,
            coef_q24,
            target_total,
            eps_scaled,
        }
    }

    /// Reset KV caches for new sequence
    pub fn reset(&mut self) {
        let config = &self.weights.config;
        for cache in &mut self.kv_caches {
            *cache = KVCache::new(1, config.max_seq_len, config.head_dim);
        }
    }

    /// Forward pass for a single token, returns logits
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<i32> {
        let config = &self.weights.config;
        let d = config.d_model;
        let d_ff = config.d_ff;
        let n_heads = config.n_heads;
        let head_dim = config.head_dim;
        let kv_dim = config.n_kv_heads * head_dim;

        // Embedding lookup (int8) -> promote to int32 for residual stream
        // CUDA keeps residual as int32 throughout, only clamping when feeding into norms
        let emb_start = (token as usize) * d;
        let embedding = &self.weights.embedding[emb_start..emb_start + d];
        let mut x: Vec<i32> = embedding.iter().map(|&v| v as i32).collect();

        // Process each layer
        // Architecture: sandwich norms (pre + post norm for each sublayer)
        // CUDA: attn_out = attn(norm1(x)); x = x + norm3(attn_out)
        //       mlp_out = mlp(norm2(x)); x = x + norm4(mlp_out)
        // Mapping: norm1=attn_norm, norm2=post_attn_norm, norm3=ffn_norm, norm4=post_ffn_norm
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            // Pre-attention norm (norm1 = attn_norm)
            // Clamp int32 residual to int8 before norm (matches CUDA RMSNorm.forward)
            let x_i8: Vec<i8> = x.iter().map(|&v| clamp_i8(v)).collect();
            let normed = rmsnorm_i8_to_i8(&x_i8, &layer.attn_norm, d, self.eps_scaled);

            // QKV projection (combined)
            // qkv_weight shape: [d_model + 2*kv_dim, d_model]
            let qkv_i32 = linear_i8_to_i32(&normed, &layer.qkv_weight, 1, d, d + 2 * kv_dim);

            // Split Q, K, V
            let q_i32: Vec<i32> = qkv_i32[..d].to_vec();
            let k_i32: Vec<i32> = qkv_i32[d..d + kv_dim].to_vec();
            let v_i32: Vec<i32> = qkv_i32[d + kv_dim..d + 2 * kv_dim].to_vec();

            // Requantize to int8
            let q = requantize_i32_to_i8(&q_i32, layer.qkv_mult, layer.qkv_shift);
            let k = requantize_i32_to_i8(&k_i32, layer.qkv_mult, layer.qkv_shift);
            let v = requantize_i32_to_i8(&v_i32, layer.qkv_mult, layer.qkv_shift);

            // Concatenate QKV for attention [d + 2*kv_dim]
            // Note: No Q normalization - CUDA applies norm3 to attention OUTPUT
            let mut qkv = Vec::with_capacity(d + 2 * kv_dim);
            qkv.extend_from_slice(&q);
            qkv.extend_from_slice(&k);
            qkv.extend_from_slice(&v);

            // Use score_mult directly (already i32)
            let score_mul_q15: &[i32] = &layer.score_mult;

            // Attention
            let attn_out = gqa_attention_mqa_i8(
                &qkv,
                &mut self.kv_caches[layer_idx],
                pos,
                n_heads,
                &score_mul_q15,
                &self.rope_lut,
                &self.exp_lut,
            );

            // Output projection
            let o_i32 = linear_i8_to_i32(&attn_out, &layer.out_weight, 1, d, d);
            let o = requantize_i32_to_i8(&o_i32, layer.out_mult, layer.out_shift);

            // Post-attention norm (norm3 = ffn_norm in conversion naming)
            // Applied to attention output BEFORE residual add
            let o_normed = rmsnorm_i8_to_i8(&o, &layer.ffn_norm, d, self.eps_scaled);

            // Residual add - keep as int32 (no clamp, matches CUDA residual_add_i8)
            for i in 0..d {
                x[i] = x[i] + (o_normed[i] as i32);
            }

            // Pre-FFN norm (norm2 = post_attn_norm in conversion naming)
            // Clamp int32 residual to int8 before norm (matches CUDA RMSNorm.forward)
            let x_i8_ffn: Vec<i8> = x.iter().map(|&v| clamp_i8(v)).collect();
            let normed_ffn = rmsnorm_i8_to_i8(&x_i8_ffn, &layer.post_attn_norm, d, self.eps_scaled);

            // FFN: combined gate+up projection
            // up_weight shape: [2*d_ff, d_model]
            let up_i32 = linear_i8_to_i32(&normed_ffn, &layer.up_weight, 1, d, 2 * d_ff);

            // Split gate and up
            let gate_i32: Vec<i32> = up_i32[..d_ff].to_vec();
            let up_i32_split: Vec<i32> = up_i32[d_ff..2 * d_ff].to_vec();

            let gate = requantize_i32_to_i8(&gate_i32, layer.up_mult, layer.up_shift);
            let up = requantize_i32_to_i8(&up_i32_split, layer.up_mult, layer.up_shift);

            // ReGLU activation
            let mut gate_up = Vec::with_capacity(2 * d_ff);
            gate_up.extend_from_slice(&gate);
            gate_up.extend_from_slice(&up);
            let gated = reglu_i8(&gate_up, d_ff);

            // Down projection
            let down_i32 = linear_i8_to_i32(&gated, &layer.down_weight, 1, d_ff, d);
            let down = requantize_i32_to_i8(&down_i32, layer.down_mult, layer.down_shift);

            // Post-FFN norm (norm4 = post_ffn_norm)
            let down_normed = rmsnorm_i8_to_i8(&down, &layer.post_ffn_norm, d, self.eps_scaled);

            // Residual add - keep as int32 (no clamp, matches CUDA residual_add_i8)
            for i in 0..d {
                x[i] = x[i] + (down_normed[i] as i32);
            }
        }

        // Final norm (int8 Q0.7 weights)
        // Clamp int32 residual to int8 before norm
        let x_i8_final: Vec<i8> = x.iter().map(|&v| clamp_i8(v)).collect();
        let final_normed = rmsnorm_i8_to_i8(&x_i8_final, &self.weights.final_norm, d, self.eps_scaled);

        // LM head - returns int32 logits (not requantized for softmax)
        let logits = linear_i8_to_i32(&final_normed, &self.weights.head_weight, 1, d, config.vocab_size);

        logits
    }

    /// Get cumulative frequencies from logits
    pub fn get_cumfreqs(&self, logits: &[i32]) -> (Vec<u32>, u32) {
        let cumfreqs = build_cumfreqs(logits, &self.exp_lut, self.coef_q24, self.target_total);
        (cumfreqs, self.target_total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 1024);
        assert_eq!(config.n_layers, 1);
        assert_eq!(config.n_heads, 6);
    }

    #[test]
    fn test_requantize() {
        let x = vec![8192i32, 4096, -8192];
        let result = requantize_i32_to_i8(&x, 1, 7);
        assert_eq!(result[0], 64);
        assert_eq!(result[1], 32);
        assert_eq!(result[2], -64);
    }
}
