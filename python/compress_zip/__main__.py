"""
Neural compression CLI.

Compresses and decompresses text files using a small transformer model.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import tiktoken

from .arith_coder import ArithEncoder, ArithDecoder
from .file_format import (
    CompressedFile,
    Codec,
    TOKENS_PER_CHUNK,
)
from .model import Model, ModelWeights


def hash_model_path(path: Path) -> int:
    """Hash model path for model_id_hash field."""
    return int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16)


def load_embedded_tokenizer(weights: ModelWeights) -> tiktoken.Encoding:
    """Load tokenizer embedded in model weights."""
    if not weights.tokenizer_json:
        raise ValueError("Model does not have embedded tokenizer")

    tok_data = json.loads(weights.tokenizer_json)

    # Convert mergeable_ranks keys to bytes
    # Keys are stored as strings (latin-1 encoded byte sequences)
    mergeable_ranks = {}
    for k, v in tok_data["mergeable_ranks"].items():
        # Keys are stored as strings representing byte sequences
        # Use latin-1 encoding which maps bytes 0-255 to unicode 0-255
        key_bytes = k.encode("latin-1")
        mergeable_ranks[key_bytes] = v

    # Create tiktoken Encoding
    return tiktoken.Encoding(
        name=tok_data.get("name", "embedded"),
        pat_str=tok_data["pat_str"],
        mergeable_ranks=mergeable_ranks,
        special_tokens=tok_data.get("special_tokens", {}),
        explicit_n_vocab=tok_data.get("explicit_n_vocab"),
    )


def load_tokenizer(tokenizer_name: str, weights: ModelWeights = None) -> tiktoken.Encoding:
    """Load tokenizer by name or from model."""
    if tokenizer_name == "embedded":
        if weights is None:
            raise ValueError("Cannot load embedded tokenizer without model weights")
        return load_embedded_tokenizer(weights)
    else:
        return tiktoken.get_encoding(tokenizer_name)


def compress(
    model_path: Path,
    tokenizer_name: str,
    input_path: Path,
    output_path: Path,
    codec_str: str,
) -> None:
    """Compress a text file."""
    total_start = time.time()

    # Parse codec
    codec_str = codec_str.lower()
    if codec_str == "zstd":
        codec = Codec.ZSTD
    elif codec_str == "brotli":
        codec = Codec.BROTLI
    else:
        print(f"Error: Unknown codec: {codec_str}", file=sys.stderr)
        sys.exit(1)

    # Load model first (needed for embedded tokenizer)
    print(f"Loading model from {model_path}...")
    try:
        weights = ModelWeights.load(model_path)
        model = Model(weights)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer '{tokenizer_name}'...")
    try:
        tokenizer = load_tokenizer(tokenizer_name, weights)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # Read input text
    print(f"Reading input from {input_path}...")
    input_text = input_path.read_text(encoding="utf-8")
    input_bytes = len(input_text.encode("utf-8"))

    # Tokenize
    print("Tokenizing...")
    tokens = tokenizer.encode(input_text)
    num_tokens = len(tokens)
    print(f"  {input_bytes} bytes -> {num_tokens} tokens")

    # Compress
    print("Compressing...")
    encode_start = time.time()

    model_id_hash = hash_model_path(model_path)
    cf = CompressedFile.create(
        model_id_hash=model_id_hash,
        tokenizer_id=0,
        codec=codec,
    )

    # Process in chunks of TOKENS_PER_CHUNK
    chunk_size = TOKENS_PER_CHUNK
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_tokens)
        chunk_tokens = tokens[chunk_start:chunk_end]
        is_last = chunk_idx == num_chunks - 1

        # Reset model (KV cache) for each chunk - chunks are independent
        model.reset()

        encoder = ArithEncoder()

        # Encode each token in the chunk
        for i, token in enumerate(chunk_tokens):
            # Position is within the chunk (0 to chunk_size-1), not global
            pos = i

            # Get model prediction (logits)
            # First token in chunk uses BOS (0), otherwise previous token in chunk
            prev_token = 0 if i == 0 else chunk_tokens[i - 1]
            logits = model.forward(prev_token, pos)

            # Convert to cumulative frequencies
            cumfreqs, total = model.get_cumfreqs(logits)

            # Encode the actual token
            encoder.encode_symbol(cumfreqs, token, total)

        chunk_data = encoder.finish()
        last_chunk_tokens = len(chunk_tokens) if is_last else TOKENS_PER_CHUNK

        cf.add_chunk(chunk_data, is_last=is_last, last_chunk_tokens=last_chunk_tokens)

        # Progress
        if (chunk_idx + 1) % 10 == 0 or is_last:
            print(f"  Chunk {chunk_idx + 1}/{num_chunks}")

    encode_time = time.time() - encode_start

    # Finalize and write output
    print(f"Writing output to {output_path}...")
    cf.finalize(uncompressed_text_len=input_bytes)
    cf.write(output_path)

    output_bytes = output_path.stat().st_size

    total_time = time.time() - total_start

    # Report stats
    print()
    print("=== Compression Results ===")
    print(f"Input:  {input_bytes} bytes ({num_tokens} tokens)")
    print(f"Output: {output_bytes} bytes")
    print(f"Ratio:  {input_bytes / output_bytes:.2f}x ({100.0 * output_bytes / input_bytes:.2f}%)")
    print(f"Bits per token: {8.0 * output_bytes / num_tokens:.2f}")
    print()
    print(f"Encode time: {encode_time:.2f}s ({num_tokens / encode_time:.1f} tokens/sec)")
    print(f"Total time:  {total_time:.2f}s")


def decompress(
    model_path: Path,
    tokenizer_name: str,
    input_path: Path,
    output_path: Path,
) -> None:
    """Decompress a compressed file."""
    total_start = time.time()

    # Load model first (needed for embedded tokenizer)
    print(f"Loading model from {model_path}...")
    try:
        weights = ModelWeights.load(model_path)
        model = Model(weights)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer '{tokenizer_name}'...")
    try:
        tokenizer = load_tokenizer(tokenizer_name, weights)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # Read compressed file
    print(f"Reading compressed file from {input_path}...")
    try:
        cf = CompressedFile.read(input_path)
    except Exception as e:
        print(f"Error reading compressed file: {e}", file=sys.stderr)
        sys.exit(1)

    num_tokens = cf.get_total_tokens()
    print(f"  {cf.inner_header.chunk_count} chunks, {num_tokens} tokens total")

    # Decompress
    print("Decompressing...")
    decode_start = time.time()

    all_tokens = []

    for chunk_idx, chunk_data in enumerate(cf.chunks):
        chunk_token_count = cf.get_chunk_tokens(chunk_idx)
        decoder = ArithDecoder(chunk_data)

        # Reset model (KV cache) for each chunk - chunks are independent
        model.reset()

        # Tokens decoded in this chunk (for prev_token lookup)
        chunk_decoded = []

        for i in range(chunk_token_count):
            # Position is within the chunk (0 to chunk_size-1), not global
            pos = i

            # Get model prediction
            # First token in chunk uses BOS (0), otherwise previous token in this chunk
            prev_token = 0 if i == 0 else chunk_decoded[i - 1]
            logits = model.forward(prev_token, pos)

            # Convert to cumulative frequencies
            cumfreqs, total = model.get_cumfreqs(logits)

            # Decode token
            token = decoder.decode_symbol(cumfreqs, total)
            chunk_decoded.append(token)

        # Add all tokens from this chunk to the full list
        all_tokens.extend(chunk_decoded)

        # Progress
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == cf.inner_header.chunk_count - 1:
            print(f"  Chunk {chunk_idx + 1}/{cf.inner_header.chunk_count}")

    decode_time = time.time() - decode_start

    # Detokenize
    print("Detokenizing...")
    output_text = tokenizer.decode(all_tokens)

    # Write output
    print(f"Writing output to {output_path}...")
    output_path.write_text(output_text, encoding="utf-8")

    output_bytes = len(output_text.encode("utf-8"))

    total_time = time.time() - total_start

    # Report stats
    print()
    print("=== Decompression Results ===")
    print(f"Tokens: {num_tokens}")
    print(f"Output: {output_bytes} bytes")
    print()
    print(f"Decode time: {decode_time:.2f}s ({num_tokens / decode_time:.1f} tokens/sec)")
    print(f"Total time:  {total_time:.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Neural text compression using a small transformer model"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress a text file")
    compress_parser.add_argument("-m", "--model", type=Path, required=True,
                                  help="Path to the model checkpoint (safetensors)")
    compress_parser.add_argument("-t", "--tokenizer", type=str, default="embedded",
                                  help="Tokenizer: 'embedded' (from model) or tiktoken name (default: embedded)")
    compress_parser.add_argument("-i", "--input", type=Path, required=True,
                                  help="Input text file")
    compress_parser.add_argument("-o", "--output", type=Path, required=True,
                                  help="Output compressed file")
    compress_parser.add_argument("-c", "--codec", type=str, default="zstd",
                                  choices=["zstd", "brotli"],
                                  help="Compression codec (default: zstd)")

    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a compressed file")
    decompress_parser.add_argument("-m", "--model", type=Path, required=True,
                                    help="Path to the model checkpoint (safetensors)")
    decompress_parser.add_argument("-t", "--tokenizer", type=str, default="embedded",
                                    help="Tokenizer: 'embedded' (from model) or tiktoken name (default: embedded)")
    decompress_parser.add_argument("-i", "--input", type=Path, required=True,
                                    help="Input compressed file")
    decompress_parser.add_argument("-o", "--output", type=Path, required=True,
                                    help="Output text file")

    args = parser.parse_args()

    if args.command == "compress":
        compress(args.model, args.tokenizer, args.input, args.output, args.codec)
    elif args.command == "decompress":
        decompress(args.model, args.tokenizer, args.input, args.output)


if __name__ == "__main__":
    main()
