"""
CZIPv1 file format for neural compression.

Outer envelope structure:
- 6 bytes: magic "CZIPv1"
- 1 byte: flags (bit0=is_multifile, bit1=training_marker, bit2=reserved)
- 1 byte: codec_id (0=zstd, 1=brotli)
- 4 bytes: uncompressed_len_bytes (u32 LE) - original text byte length
- 4 bytes: header_len_bytes (u32 LE) - inner header length before compression
- payload: compressed bytes of [inner_header || chunk_data]

Inner header structure:
- u32: model_id_hash
- u32: chunk_count
- u16: last_chunk_tokens (1..64)
- u8: tokenizer_id
- u8: reserved
- u16[chunk_count]: chunk_byte_len array
- [chunk_data]: concatenated chunk payloads
"""

import struct
import zlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
from enum import IntEnum

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


# Magic number for file format identification
CZIP_MAGIC = b'CZIPv1'  # 6 bytes
OUTER_HEADER_SIZE = 16  # Fixed outer header size

# Chunk configuration
TOKENS_PER_CHUNK = 64  # All chunks except last have exactly 64 tokens


class Codec(IntEnum):
    """Compression codec identifiers."""
    ZSTD = 0
    BROTLI = 1


class Flags(IntEnum):
    """Flag bit positions."""
    IS_MULTIFILE = 0
    TRAINING_MARKER = 1
    RESERVED = 2


@dataclass
class CZIPv1OuterHeader:
    """
    CZIPv1 outer envelope header.

    Layout (16 bytes):
    - magic: 6 bytes ("CZIPv1")
    - flags: 1 byte (bitfield)
    - codec_id: 1 byte (0=zstd, 1=brotli)
    - uncompressed_len: 4 bytes (u32 LE, original text byte length)
    - header_len: 4 bytes (u32 LE, inner header length before compression)
    """
    magic: bytes = CZIP_MAGIC
    flags: int = 0
    codec_id: int = Codec.ZSTD
    uncompressed_len: int = 0  # Original text byte length
    header_len: int = 0  # Inner header length before compression

    def is_multifile(self) -> bool:
        return bool(self.flags & (1 << Flags.IS_MULTIFILE))

    def set_multifile(self, value: bool):
        if value:
            self.flags |= (1 << Flags.IS_MULTIFILE)
        else:
            self.flags &= ~(1 << Flags.IS_MULTIFILE)

    def to_bytes(self) -> bytes:
        """Serialize outer header to bytes."""
        return struct.pack(
            '<6sBBII',
            self.magic,
            self.flags,
            self.codec_id,
            self.uncompressed_len,
            self.header_len,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "CZIPv1OuterHeader":
        """Deserialize outer header from bytes."""
        if len(data) < OUTER_HEADER_SIZE:
            raise ValueError(f"Outer header too short: {len(data)} < {OUTER_HEADER_SIZE}")

        magic, flags, codec_id, uncompressed_len, header_len = struct.unpack(
            '<6sBBII',
            data[:OUTER_HEADER_SIZE]
        )

        if magic != CZIP_MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {CZIP_MAGIC!r}")

        return cls(
            magic=magic,
            flags=flags,
            codec_id=codec_id,
            uncompressed_len=uncompressed_len,
            header_len=header_len,
        )


@dataclass
class CZIPv1InnerHeader:
    """
    CZIPv1 inner header (before outer compression).

    Layout:
    - model_id_hash: 4 bytes (u32 LE, for mismatch detection, 0 if unused)
    - chunk_count: 4 bytes (u32 LE)
    - last_chunk_tokens: 2 bytes (u16 LE, 1..64)
    - tokenizer_id: 1 byte (u8, 0 for default)
    - reserved: 1 byte (u8)
    - chunk_byte_lens: chunk_count * 2 bytes (u16 LE each)
    """
    model_id_hash: int = 0
    chunk_count: int = 0
    last_chunk_tokens: int = TOKENS_PER_CHUNK
    tokenizer_id: int = 0
    reserved: int = 0
    chunk_byte_lens: List[int] = field(default_factory=list)

    def header_size(self) -> int:
        """Get inner header size in bytes (excluding chunk data)."""
        return 12 + 2 * self.chunk_count

    def to_bytes(self) -> bytes:
        """Serialize inner header to bytes."""
        buf = struct.pack(
            '<IIHBB',
            self.model_id_hash,
            self.chunk_count,
            self.last_chunk_tokens,
            self.tokenizer_id,
            self.reserved,
        )
        # Append chunk byte lengths
        for byte_len in self.chunk_byte_lens:
            buf += struct.pack('<H', byte_len)
        return buf

    @classmethod
    def from_bytes(cls, data: bytes) -> "CZIPv1InnerHeader":
        """Deserialize inner header from bytes."""
        if len(data) < 12:
            raise ValueError(f"Inner header too short: {len(data)} < 12")

        model_id_hash, chunk_count, last_chunk_tokens, tokenizer_id, reserved = struct.unpack(
            '<IIHBB',
            data[:12]
        )

        expected_len = 12 + 2 * chunk_count
        if len(data) < expected_len:
            raise ValueError(f"Inner header too short for {chunk_count} chunks")

        chunk_byte_lens = []
        offset = 12
        for _ in range(chunk_count):
            byte_len = struct.unpack('<H', data[offset:offset + 2])[0]
            chunk_byte_lens.append(byte_len)
            offset += 2

        return cls(
            model_id_hash=model_id_hash,
            chunk_count=chunk_count,
            last_chunk_tokens=last_chunk_tokens,
            tokenizer_id=tokenizer_id,
            reserved=reserved,
            chunk_byte_lens=chunk_byte_lens,
        )


def compress_payload(data: bytes, codec: Codec, level: Optional[int] = None) -> bytes:
    """Compress payload using specified codec."""
    if codec == Codec.ZSTD:
        if not HAS_ZSTD:
            raise RuntimeError("zstandard library not installed")
        lvl = level if level is not None else 22
        cctx = zstd.ZstdCompressor(level=lvl)
        return cctx.compress(data)
    elif codec == Codec.BROTLI:
        if not HAS_BROTLI:
            raise RuntimeError("brotli library not installed")
        lvl = level if level is not None else 11
        return brotli.compress(data, quality=lvl)
    else:
        raise ValueError(f"Unknown codec: {codec}")


def decompress_payload(data: bytes, codec: Codec) -> bytes:
    """Decompress payload using specified codec."""
    if codec == Codec.ZSTD:
        if not HAS_ZSTD:
            raise RuntimeError("zstandard library not installed")
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    elif codec == Codec.BROTLI:
        if not HAS_BROTLI:
            raise RuntimeError("brotli library not installed")
        return brotli.decompress(data)
    else:
        raise ValueError(f"Unknown codec: {codec}")


class CompressedFile:
    """
    Read/write CZIPv1 compressed files.
    """

    def __init__(self):
        self.outer_header: CZIPv1OuterHeader = CZIPv1OuterHeader()
        self.inner_header: CZIPv1InnerHeader = CZIPv1InnerHeader()
        self.chunks: List[bytes] = []  # Raw arithmetic-coded chunk data

    @classmethod
    def create(
        cls,
        model_id_hash: int = 0,
        tokenizer_id: int = 0,
        codec: Codec = Codec.ZSTD,
        is_multifile: bool = False,
    ) -> "CompressedFile":
        """Create a new compressed file."""
        cf = cls()
        cf.outer_header.codec_id = codec
        cf.outer_header.set_multifile(is_multifile)
        cf.inner_header.model_id_hash = model_id_hash
        cf.inner_header.tokenizer_id = tokenizer_id
        return cf

    def add_chunk(self, compressed_data: bytes, is_last: bool = False, last_chunk_tokens: int = TOKENS_PER_CHUNK):
        """
        Add a compressed chunk.

        Args:
            compressed_data: Arithmetic-coded bytes for this chunk
            is_last: Whether this is the last chunk
            last_chunk_tokens: Number of tokens in last chunk (1..64)
        """
        if len(compressed_data) > 65535:
            raise ValueError(f"Chunk too large: {len(compressed_data)} > 65535 bytes")

        self.chunks.append(compressed_data)
        self.inner_header.chunk_byte_lens.append(len(compressed_data))
        self.inner_header.chunk_count = len(self.chunks)

        if is_last:
            self.inner_header.last_chunk_tokens = last_chunk_tokens

    def finalize(self, uncompressed_text_len: int, compression_level: Optional[int] = None):
        """
        Finalize the file for writing.

        Args:
            uncompressed_text_len: Original text length in bytes (before tokenization)
            compression_level: Optional compression level override
        """
        self.outer_header.uncompressed_len = uncompressed_text_len
        self.outer_header.header_len = self.inner_header.header_size()

    def _build_payload(self) -> bytes:
        """Build the uncompressed payload (inner_header || chunk_data)."""
        payload = self.inner_header.to_bytes()
        for chunk in self.chunks:
            payload += chunk
        return payload

    def write(self, path: Path | str, compression_level: Optional[int] = None):
        """Write compressed file to disk."""
        path = Path(path)

        # Build and compress payload
        payload = self._build_payload()
        compressed_payload = compress_payload(
            payload,
            Codec(self.outer_header.codec_id),
            compression_level
        )

        # Update header with final sizes
        self.outer_header.header_len = self.inner_header.header_size()

        with open(path, 'wb') as f:
            f.write(self.outer_header.to_bytes())
            f.write(compressed_payload)

    def to_bytes(self, compression_level: Optional[int] = None) -> bytes:
        """Serialize to bytes."""
        payload = self._build_payload()
        compressed_payload = compress_payload(
            payload,
            Codec(self.outer_header.codec_id),
            compression_level
        )
        return self.outer_header.to_bytes() + compressed_payload

    @classmethod
    def read(cls, path: Path | str) -> "CompressedFile":
        """Read compressed file from disk."""
        path = Path(path)

        with open(path, 'rb') as f:
            data = f.read()

        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompressedFile":
        """Deserialize from bytes."""
        cf = cls()

        # Parse outer header
        cf.outer_header = CZIPv1OuterHeader.from_bytes(data[:OUTER_HEADER_SIZE])

        # Decompress payload
        compressed_payload = data[OUTER_HEADER_SIZE:]
        payload = decompress_payload(compressed_payload, Codec(cf.outer_header.codec_id))

        # Parse inner header
        cf.inner_header = CZIPv1InnerHeader.from_bytes(payload)

        # Extract chunks
        offset = cf.inner_header.header_size()
        cf.chunks = []
        for byte_len in cf.inner_header.chunk_byte_lens:
            chunk_data = payload[offset:offset + byte_len]
            if len(chunk_data) != byte_len:
                raise ValueError(f"Chunk data truncated: {len(chunk_data)} < {byte_len}")
            cf.chunks.append(chunk_data)
            offset += byte_len

        return cf

    def get_total_tokens(self) -> int:
        """Get total number of tokens across all chunks."""
        if self.inner_header.chunk_count == 0:
            return 0
        # All chunks except last have TOKENS_PER_CHUNK tokens
        full_chunks = self.inner_header.chunk_count - 1
        return full_chunks * TOKENS_PER_CHUNK + self.inner_header.last_chunk_tokens

    def get_chunk_tokens(self, chunk_idx: int) -> int:
        """Get number of tokens in a specific chunk."""
        if chunk_idx < 0 or chunk_idx >= self.inner_header.chunk_count:
            raise IndexError(f"Chunk index {chunk_idx} out of range")
        if chunk_idx == self.inner_header.chunk_count - 1:
            return self.inner_header.last_chunk_tokens
        return TOKENS_PER_CHUNK

    def get_compressed_size(self) -> int:
        """Get total compressed size in bytes (outer header + compressed payload)."""
        payload = self._build_payload()
        compressed = compress_payload(payload, Codec(self.outer_header.codec_id))
        return OUTER_HEADER_SIZE + len(compressed)


# Test cases
if __name__ == "__main__":
    import tempfile

    print("Testing CZIPv1 file format...")

    # Test outer header serialization
    outer = CZIPv1OuterHeader(
        codec_id=Codec.ZSTD,
        uncompressed_len=1000,
        header_len=20,
    )
    outer.set_multifile(True)
    outer_bytes = outer.to_bytes()
    print(f"Outer header size: {len(outer_bytes)} bytes")
    assert len(outer_bytes) == OUTER_HEADER_SIZE

    outer2 = CZIPv1OuterHeader.from_bytes(outer_bytes)
    assert outer2.codec_id == outer.codec_id
    assert outer2.uncompressed_len == outer.uncompressed_len
    assert outer2.is_multifile() == True
    print("Outer header roundtrip passed!")

    # Test inner header serialization
    inner = CZIPv1InnerHeader(
        model_id_hash=0x12345678,
        chunk_count=3,
        last_chunk_tokens=32,
        tokenizer_id=0,
        chunk_byte_lens=[100, 150, 80],
    )
    inner_bytes = inner.to_bytes()
    print(f"Inner header size: {len(inner_bytes)} bytes (expected {inner.header_size()})")
    assert len(inner_bytes) == inner.header_size()

    inner2 = CZIPv1InnerHeader.from_bytes(inner_bytes)
    assert inner2.model_id_hash == inner.model_id_hash
    assert inner2.chunk_count == inner.chunk_count
    assert inner2.last_chunk_tokens == inner.last_chunk_tokens
    assert inner2.chunk_byte_lens == inner.chunk_byte_lens
    print("Inner header roundtrip passed!")

    # Test file read/write (only if compression libs available)
    if HAS_ZSTD:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test.czip"

            # Create file
            cf = CompressedFile.create(
                model_id_hash=0xDEADBEEF,
                tokenizer_id=0,
                codec=Codec.ZSTD,
            )

            # Add some chunks
            cf.add_chunk(b'\x00' * 100)
            cf.add_chunk(b'\x11' * 150)
            cf.add_chunk(b'\x22' * 80, is_last=True, last_chunk_tokens=32)

            cf.finalize(uncompressed_text_len=1000)

            # Write
            cf.write(test_path)
            print(f"Wrote file to {test_path}")

            # Read back
            cf2 = CompressedFile.read(test_path)

            assert cf2.inner_header.model_id_hash == cf.inner_header.model_id_hash
            assert cf2.inner_header.chunk_count == cf.inner_header.chunk_count
            assert cf2.inner_header.last_chunk_tokens == cf.inner_header.last_chunk_tokens
            assert len(cf2.chunks) == len(cf.chunks)

            for i, (d1, d2) in enumerate(zip(cf.chunks, cf2.chunks)):
                assert d1 == d2, f"Chunk {i} data mismatch"

            print("File roundtrip passed!")

            # Test total tokens
            assert cf2.get_total_tokens() == 64 + 64 + 32
            print(f"Total tokens: {cf2.get_total_tokens()}")

        print("\nAll CZIPv1 file format tests passed!")
    else:
        print("\nSkipping file tests (zstandard not installed)")
        print("Install with: pip install zstandard")
