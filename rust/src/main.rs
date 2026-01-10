//! Neural compression CLI
//!
//! Compresses and decompresses text files using a small transformer model.

use clap::{Parser, Subcommand};
use compress_zip::arith_coder::{ArithDecoder, ArithEncoder};
use compress_zip::file_format::{CompressedFile, Codec, CZIPv1OuterHeader, TOKENS_PER_CHUNK};
use compress_zip::model::{Model, ModelWeights};
use compress_zip::tiktoken::{load_tiktoken_json_tokenizer, load_tiktoken_json_from_bytes};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "compress_zip")]
#[command(about = "Neural text compression using a small transformer model")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a text file
    Compress {
        /// Path to the model checkpoint (safetensors)
        #[arg(short, long)]
        model: PathBuf,

        /// Path to the tokenizer (JSON). If not provided, uses embedded tokenizer from model
        #[arg(short, long)]
        tokenizer: Option<PathBuf>,

        /// Input text file
        #[arg(short, long)]
        input: PathBuf,

        /// Output compressed file
        #[arg(short, long)]
        output: PathBuf,

        /// Compression codec (zstd or brotli)
        #[arg(short, long, default_value = "zstd")]
        codec: String,
    },
    /// Decompress a compressed file
    Decompress {
        /// Path to the model checkpoint (safetensors)
        #[arg(short, long)]
        model: PathBuf,

        /// Path to the tokenizer (JSON). If not provided, uses embedded tokenizer from model
        #[arg(short, long)]
        tokenizer: Option<PathBuf>,

        /// Input compressed file
        #[arg(short, long)]
        input: PathBuf,

        /// Output text file
        #[arg(short, long)]
        output: PathBuf,
    },
}

/// Hash model path for model_id_hash field
fn hash_model_path(path: &PathBuf) -> u32 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    (hasher.finish() & 0xFFFFFFFF) as u32
}

/// Decompress tokenizer data using the specified codec
fn decompress_tokenizer_data(data: &[u8], codec: &str) -> io::Result<Vec<u8>> {
    match codec {
        "zstd" => {
            use std::io::Read;
            let mut decoder = zstd::Decoder::new(data)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            Ok(decompressed)
        }
        "brotli" => {
            let mut decompressed = Vec::new();
            brotli::BrotliDecompress(&mut std::io::Cursor::new(data), &mut decompressed)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            Ok(decompressed)
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Unknown tokenizer codec: {}", codec),
        )),
    }
}

fn compress(
    model_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    input_path: PathBuf,
    output_path: PathBuf,
    codec_str: String,
) -> io::Result<()> {
    let total_start = Instant::now();

    // Parse codec
    let codec = match codec_str.to_lowercase().as_str() {
        "zstd" => Codec::Zstd,
        "brotli" => Codec::Brotli,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unknown codec: {}", codec_str),
            ));
        }
    };

    // Load model first (needed for embedded tokenizer)
    println!("Loading model from {:?}...", model_path);
    let weights = ModelWeights::load(&model_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // Load tokenizer (from file or embedded in model)
    let tokenizer = if let Some(ref tok_path) = tokenizer_path {
        println!("Loading tokenizer from {:?}...", tok_path);
        load_tiktoken_json_tokenizer(tok_path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
    } else {
        println!("Loading embedded tokenizer from model...");
        let tok_data = weights.tokenizer_data.as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "No embedded tokenizer in model"))?;
        // Decompress tokenizer data
        let decompressed = decompress_tokenizer_data(tok_data, &weights.tokenizer_codec)?;
        load_tiktoken_json_from_bytes(&decompressed)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
    };

    let mut model = Model::new(weights);

    // Read input text
    println!("Reading input from {:?}...", input_path);
    let input_text = fs::read_to_string(&input_path)?;
    let input_bytes = input_text.len();

    // Tokenize
    println!("Tokenizing...");
    let tokens = tokenizer.encode_with_special_tokens(&input_text);
    let num_tokens = tokens.len();
    println!("  {} bytes -> {} tokens", input_bytes, num_tokens);

    // Compress
    println!("Compressing...");
    let encode_start = Instant::now();

    let model_id_hash = hash_model_path(&model_path);
    let mut cf = CompressedFile::new(
        model_id_hash,
        0, // tokenizer_id
        codec,
        input_bytes as u32,
    );

    // Process in chunks of TOKENS_PER_CHUNK
    let chunk_size = TOKENS_PER_CHUNK as usize;
    let num_chunks = (num_tokens + chunk_size - 1) / chunk_size;

    for chunk_idx in 0..num_chunks {
        let chunk_start = chunk_idx * chunk_size;
        let chunk_end = std::cmp::min(chunk_start + chunk_size, num_tokens);
        let chunk_tokens = &tokens[chunk_start..chunk_end];
        let is_last = chunk_idx == num_chunks - 1;

        // Reset model (KV cache) for each chunk - chunks are independent
        model.reset();

        let mut encoder = ArithEncoder::new();

        // Encode each token in the chunk
        for (i, &token) in chunk_tokens.iter().enumerate() {
            // Position is within the chunk (0 to chunk_size-1), not global
            let pos = i;

            // Get model prediction (logits)
            // First token in chunk uses BOS (1022), otherwise previous token in chunk
            // BOS token 1022 matches the CUDA implementation
            let prev_token = if i == 0 { 1022 } else { chunk_tokens[i - 1] };
            let logits = model.forward(prev_token, pos);

            // Convert to cumulative frequencies
            let (cumfreqs, _total) = model.get_cumfreqs(&logits);

            // Encode the actual token
            encoder.encode_symbol(&cumfreqs, token as usize);
        }

        encoder.finish();
        let chunk_data = encoder.get_output().to_vec();
        let last_chunk_tokens = if is_last {
            chunk_tokens.len() as u16
        } else {
            TOKENS_PER_CHUNK
        };

        cf.add_chunk(chunk_data, is_last, last_chunk_tokens);
    }

    let encode_time = encode_start.elapsed();

    // Write output with compression
    println!("Writing output to {:?}...", output_path);

    let (outer_bytes, payload) = cf.to_bytes_uncompressed();

    // Compress payload with zstd (or brotli based on codec)
    let compressed_payload = match codec {
        Codec::Zstd => {
            zstd::encode_all(std::io::Cursor::new(&payload), 3)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
        }
        Codec::Brotli => {
            let mut compressed = Vec::new();
            {
                let mut encoder = brotli::CompressorWriter::new(&mut compressed, 4096, 6, 22);
                encoder.write_all(&payload)?;
            }
            compressed
        }
    };

    let mut file = fs::File::create(&output_path)?;
    file.write_all(&outer_bytes)?;
    file.write_all(&compressed_payload)?;

    let output_bytes = outer_bytes.len() + compressed_payload.len();

    let total_time = total_start.elapsed();

    // Report stats
    println!("\n=== Compression Results ===");
    println!("Input:  {} bytes ({} tokens)", input_bytes, num_tokens);
    println!("Output: {} bytes", output_bytes);
    println!("Ratio:  {:.2}x ({:.2}%)",
        input_bytes as f64 / output_bytes as f64,
        100.0 * output_bytes as f64 / input_bytes as f64
    );
    println!("Bits per token: {:.2}", 8.0 * output_bytes as f64 / num_tokens as f64);
    println!();
    println!("Encode time: {:.2}s ({:.1} tokens/sec)",
        encode_time.as_secs_f64(),
        num_tokens as f64 / encode_time.as_secs_f64()
    );
    println!("Total time:  {:.2}s", total_time.as_secs_f64());

    Ok(())
}

fn decompress(
    model_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    input_path: PathBuf,
    output_path: PathBuf,
) -> io::Result<()> {
    let total_start = Instant::now();

    // Load model first (needed for embedded tokenizer)
    println!("Loading model from {:?}...", model_path);
    let weights = ModelWeights::load(&model_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // Load tokenizer (from file or embedded in model)
    let tokenizer = if let Some(ref tok_path) = tokenizer_path {
        println!("Loading tokenizer from {:?}...", tok_path);
        load_tiktoken_json_tokenizer(tok_path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
    } else {
        println!("Loading embedded tokenizer from model...");
        let tok_data = weights.tokenizer_data.as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "No embedded tokenizer in model"))?;
        // Decompress tokenizer data
        let decompressed = decompress_tokenizer_data(tok_data, &weights.tokenizer_codec)?;
        load_tiktoken_json_from_bytes(&decompressed)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
    };

    let mut model = Model::new(weights);

    // Read compressed file
    println!("Reading compressed file from {:?}...", input_path);
    let mut file = fs::File::open(&input_path)?;

    // Read outer header
    let mut outer_buf = [0u8; 16];
    file.read_exact(&mut outer_buf)?;
    let outer_header = CZIPv1OuterHeader::from_bytes(&outer_buf)?;

    // Read compressed payload
    let mut compressed_payload = Vec::new();
    file.read_to_end(&mut compressed_payload)?;

    // Decompress payload
    let payload = match outer_header.codec_id {
        Codec::Zstd => {
            zstd::decode_all(std::io::Cursor::new(&compressed_payload))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
        }
        Codec::Brotli => {
            let mut decompressed = Vec::new();
            let mut decoder = brotli::Decompressor::new(&compressed_payload[..], 4096);
            decoder.read_to_end(&mut decompressed)?;
            decompressed
        }
    };

    let cf = CompressedFile::read_from_decompressed(outer_header, &payload)?;

    let num_tokens = cf.get_total_tokens() as usize;
    println!("  {} chunks, {} tokens total", cf.inner_header.chunk_count, num_tokens);

    // Decompress
    println!("Decompressing...");
    let decode_start = Instant::now();

    let mut all_tokens: Vec<u32> = Vec::with_capacity(num_tokens);

    for (chunk_idx, chunk) in cf.chunks.iter().enumerate() {
        let chunk_token_count = cf.get_chunk_tokens(chunk_idx) as usize;
        let mut decoder = ArithDecoder::new(chunk.data.clone());

        // Reset model (KV cache) for each chunk - chunks are independent
        model.reset();

        // Tokens decoded in this chunk (for prev_token lookup)
        let mut chunk_decoded: Vec<u32> = Vec::with_capacity(chunk_token_count);

        for i in 0..chunk_token_count {
            // Position is within the chunk (0 to chunk_size-1), not global
            let pos = i;

            // Get model prediction
            // First token in chunk uses BOS (1022), otherwise previous token in this chunk
            // BOS token 1022 matches the CUDA implementation
            let prev_token = if i == 0 { 1022 } else { chunk_decoded[i - 1] };
            let logits = model.forward(prev_token, pos);

            // Convert to cumulative frequencies
            let (cumfreqs, _total) = model.get_cumfreqs(&logits);

            // Decode token
            let token = decoder.decode_symbol(&cumfreqs) as u32;
            chunk_decoded.push(token);
        }

        // Add all tokens from this chunk to the full list
        all_tokens.extend_from_slice(&chunk_decoded);
    }

    let decode_time = decode_start.elapsed();

    // Detokenize
    println!("Detokenizing...");
    let output_bytes = tokenizer.decode_bytes(&all_tokens)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let output_text = String::from_utf8_lossy(&output_bytes);

    // Write output
    println!("Writing output to {:?}...", output_path);
    fs::write(&output_path, output_text.as_bytes())?;

    let total_time = total_start.elapsed();

    // Report stats
    println!("\n=== Decompression Results ===");
    println!("Tokens: {}", num_tokens);
    println!("Output: {} bytes", output_bytes.len());
    println!();
    println!("Decode time: {:.2}s ({:.1} tokens/sec)",
        decode_time.as_secs_f64(),
        num_tokens as f64 / decode_time.as_secs_f64()
    );
    println!("Total time:  {:.2}s", total_time.as_secs_f64());

    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Compress {
            model,
            tokenizer,
            input,
            output,
            codec,
        } => compress(model, tokenizer, input, output, codec),
        Commands::Decompress {
            model,
            tokenizer,
            input,
            output,
        } => decompress(model, tokenizer, input, output),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
