//! CZIPv1 file format for compressed data.
//!
//! Outer envelope (16 bytes + compressed payload):
//! - magic: [u8; 6] = b"CZIPv1"
//! - flags: u8 (bit0=is_multifile, bit1=training_marker, bit2=reserved)
//! - codec_id: u8 (0=zstd, 1=brotli)
//! - uncompressed_len: u32 LE (original text byte length)
//! - header_len: u32 LE (inner header length before compression)
//! - payload: compressed bytes of [inner_header || chunk_data]
//!
//! Inner header (before compression):
//! - model_id_hash: u32 LE
//! - chunk_count: u32 LE
//! - last_chunk_tokens: u16 LE (1..64)
//! - tokenizer_id: u8
//! - reserved: u8
//! - chunk_byte_lens: [u16; chunk_count] LE

use std::io::{self, Read, Write};

/// File format magic bytes
pub const MAGIC: &[u8; 6] = b"CZIPv1";

/// Outer header size in bytes
pub const OUTER_HEADER_SIZE: usize = 16;

/// Fixed tokens per chunk (except last chunk)
pub const TOKENS_PER_CHUNK: u16 = 64;

/// Compression codec identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    Zstd = 0,
    Brotli = 1,
}

impl TryFrom<u8> for Codec {
    type Error = io::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Codec::Zstd),
            1 => Ok(Codec::Brotli),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown codec: {}", value),
            )),
        }
    }
}

/// Flag bit positions
pub mod flags {
    pub const IS_MULTIFILE: u8 = 0;
    pub const TRAINING_MARKER: u8 = 1;
    pub const RESERVED: u8 = 2;
}

/// CZIPv1 outer envelope header.
#[derive(Debug, Clone)]
pub struct CZIPv1OuterHeader {
    pub magic: [u8; 6],
    pub flags: u8,
    pub codec_id: Codec,
    pub uncompressed_len: u32,  // Original text byte length
    pub header_len: u32,        // Inner header length before compression
}

impl CZIPv1OuterHeader {
    /// Create a new outer header.
    pub fn new(codec: Codec, uncompressed_len: u32) -> Self {
        Self {
            magic: *MAGIC,
            flags: 0,
            codec_id: codec,
            uncompressed_len,
            header_len: 0,
        }
    }

    /// Check if multifile flag is set.
    pub fn is_multifile(&self) -> bool {
        (self.flags & (1 << flags::IS_MULTIFILE)) != 0
    }

    /// Set the multifile flag.
    pub fn set_multifile(&mut self, value: bool) {
        if value {
            self.flags |= 1 << flags::IS_MULTIFILE;
        } else {
            self.flags &= !(1 << flags::IS_MULTIFILE);
        }
    }

    /// Read header from bytes.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < OUTER_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Outer header too short",
            ));
        }

        let mut magic = [0u8; 6];
        magic.copy_from_slice(&data[0..6]);

        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid magic bytes: {:?}", magic),
            ));
        }

        let flags = data[6];
        let codec_id = Codec::try_from(data[7])?;
        let uncompressed_len = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let header_len = u32::from_le_bytes(data[12..16].try_into().unwrap());

        Ok(Self {
            magic,
            flags,
            codec_id,
            uncompressed_len,
            header_len,
        })
    }

    /// Write header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(OUTER_HEADER_SIZE);
        buf.extend_from_slice(&self.magic);
        buf.push(self.flags);
        buf.push(self.codec_id as u8);
        buf.extend_from_slice(&self.uncompressed_len.to_le_bytes());
        buf.extend_from_slice(&self.header_len.to_le_bytes());
        buf
    }
}

/// CZIPv1 inner header (before outer compression).
#[derive(Debug, Clone)]
pub struct CZIPv1InnerHeader {
    pub model_id_hash: u32,
    pub chunk_count: u32,
    pub last_chunk_tokens: u16,
    pub tokenizer_id: u8,
    pub reserved: u8,
    pub chunk_byte_lens: Vec<u16>,
}

impl CZIPv1InnerHeader {
    /// Create a new inner header.
    pub fn new(model_id_hash: u32, tokenizer_id: u8) -> Self {
        Self {
            model_id_hash,
            chunk_count: 0,
            last_chunk_tokens: TOKENS_PER_CHUNK,
            tokenizer_id,
            reserved: 0,
            chunk_byte_lens: Vec::new(),
        }
    }

    /// Get inner header size in bytes (excluding chunk data).
    pub fn header_size(&self) -> usize {
        12 + 2 * self.chunk_count as usize
    }

    /// Read header from bytes.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Inner header too short",
            ));
        }

        let model_id_hash = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let chunk_count = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let last_chunk_tokens = u16::from_le_bytes(data[8..10].try_into().unwrap());
        let tokenizer_id = data[10];
        let reserved = data[11];

        let expected_len = 12 + 2 * chunk_count as usize;
        if data.len() < expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Inner header too short for {} chunks", chunk_count),
            ));
        }

        let mut chunk_byte_lens = Vec::with_capacity(chunk_count as usize);
        let mut offset = 12;
        for _ in 0..chunk_count {
            let byte_len = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
            chunk_byte_lens.push(byte_len);
            offset += 2;
        }

        Ok(Self {
            model_id_hash,
            chunk_count,
            last_chunk_tokens,
            tokenizer_id,
            reserved,
            chunk_byte_lens,
        })
    }

    /// Write header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.header_size());
        buf.extend_from_slice(&self.model_id_hash.to_le_bytes());
        buf.extend_from_slice(&self.chunk_count.to_le_bytes());
        buf.extend_from_slice(&self.last_chunk_tokens.to_le_bytes());
        buf.push(self.tokenizer_id);
        buf.push(self.reserved);
        for &byte_len in &self.chunk_byte_lens {
            buf.extend_from_slice(&byte_len.to_le_bytes());
        }
        buf
    }
}

/// A single encoded chunk.
#[derive(Debug, Clone)]
pub struct EncodedChunk {
    pub data: Vec<u8>,
}

/// Complete compressed file (uncompressed representation).
#[derive(Debug, Clone)]
pub struct CompressedFile {
    pub outer_header: CZIPv1OuterHeader,
    pub inner_header: CZIPv1InnerHeader,
    pub chunks: Vec<EncodedChunk>,
}

impl CompressedFile {
    /// Create a new compressed file.
    pub fn new(
        model_id_hash: u32,
        tokenizer_id: u8,
        codec: Codec,
        uncompressed_text_len: u32,
    ) -> Self {
        Self {
            outer_header: CZIPv1OuterHeader::new(codec, uncompressed_text_len),
            inner_header: CZIPv1InnerHeader::new(model_id_hash, tokenizer_id),
            chunks: Vec::new(),
        }
    }

    /// Add a chunk.
    pub fn add_chunk(&mut self, data: Vec<u8>, is_last: bool, last_chunk_tokens: u16) {
        assert!(data.len() <= u16::MAX as usize, "Chunk too large");

        self.inner_header.chunk_byte_lens.push(data.len() as u16);
        self.inner_header.chunk_count = self.inner_header.chunk_byte_lens.len() as u32;

        if is_last {
            self.inner_header.last_chunk_tokens = last_chunk_tokens;
        }

        self.chunks.push(EncodedChunk { data });
    }

    /// Build the uncompressed payload (inner_header || chunk_data).
    fn build_payload(&self) -> Vec<u8> {
        let mut payload = self.inner_header.to_bytes();
        for chunk in &self.chunks {
            payload.extend_from_slice(&chunk.data);
        }
        payload
    }

    /// Get total tokens across all chunks.
    pub fn get_total_tokens(&self) -> u32 {
        if self.inner_header.chunk_count == 0 {
            return 0;
        }
        let full_chunks = self.inner_header.chunk_count - 1;
        full_chunks * TOKENS_PER_CHUNK as u32 + self.inner_header.last_chunk_tokens as u32
    }

    /// Get token count for a specific chunk.
    pub fn get_chunk_tokens(&self, chunk_idx: usize) -> u16 {
        if chunk_idx == self.inner_header.chunk_count as usize - 1 {
            self.inner_header.last_chunk_tokens
        } else {
            TOKENS_PER_CHUNK
        }
    }

    /// Read from bytes (requires decompression).
    ///
    /// Note: This expects already-decompressed payload bytes after the outer header.
    /// For full file reading with compression, use `read_from_compressed`.
    pub fn read_from_decompressed(outer_header: CZIPv1OuterHeader, payload: &[u8]) -> io::Result<Self> {
        let inner_header = CZIPv1InnerHeader::from_bytes(payload)?;

        let mut chunks = Vec::with_capacity(inner_header.chunk_count as usize);
        let mut offset = inner_header.header_size();

        for &byte_len in &inner_header.chunk_byte_lens {
            let end = offset + byte_len as usize;
            if payload.len() < end {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Chunk data truncated",
                ));
            }
            chunks.push(EncodedChunk {
                data: payload[offset..end].to_vec(),
            });
            offset = end;
        }

        Ok(Self {
            outer_header,
            inner_header,
            chunks,
        })
    }

    /// Write to bytes (uncompressed payload, for external compression).
    ///
    /// Returns (outer_header_bytes, payload_bytes) for external compression.
    pub fn to_bytes_uncompressed(&self) -> (Vec<u8>, Vec<u8>) {
        let mut outer = self.outer_header.clone();
        outer.header_len = self.inner_header.header_size() as u32;

        (outer.to_bytes(), self.build_payload())
    }

    /// Write to a writer (without compression - for testing).
    ///
    /// Note: In production, the payload should be compressed with zstd or brotli.
    pub fn write_to_uncompressed<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let (outer_bytes, payload) = self.to_bytes_uncompressed();
        writer.write_all(&outer_bytes)?;
        writer.write_all(&payload)?;
        Ok(())
    }

    /// Read from a reader (without decompression - for testing).
    ///
    /// Note: In production, the payload should be decompressed first.
    pub fn read_from_uncompressed<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut outer_buf = [0u8; OUTER_HEADER_SIZE];
        reader.read_exact(&mut outer_buf)?;
        let outer_header = CZIPv1OuterHeader::from_bytes(&outer_buf)?;

        let mut payload = Vec::new();
        reader.read_to_end(&mut payload)?;

        Self::read_from_decompressed(outer_header, &payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_outer_header_roundtrip() {
        let mut header = CZIPv1OuterHeader::new(Codec::Zstd, 1000);
        header.set_multifile(true);
        header.header_len = 20;

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), OUTER_HEADER_SIZE);

        let parsed = CZIPv1OuterHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.magic, *MAGIC);
        assert_eq!(parsed.codec_id, Codec::Zstd);
        assert_eq!(parsed.uncompressed_len, 1000);
        assert_eq!(parsed.header_len, 20);
        assert!(parsed.is_multifile());
    }

    #[test]
    fn test_inner_header_roundtrip() {
        let mut header = CZIPv1InnerHeader::new(0x12345678, 0);
        header.chunk_count = 3;
        header.last_chunk_tokens = 32;
        header.chunk_byte_lens = vec![100, 150, 80];

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), header.header_size());

        let parsed = CZIPv1InnerHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.model_id_hash, 0x12345678);
        assert_eq!(parsed.chunk_count, 3);
        assert_eq!(parsed.last_chunk_tokens, 32);
        assert_eq!(parsed.chunk_byte_lens, vec![100, 150, 80]);
    }

    #[test]
    fn test_header_roundtrip() {
        // Legacy test - using new format
        let mut file = CompressedFile::new(0xABCDEF00, 0, Codec::Zstd, 5000);
        file.add_chunk(vec![1, 2, 3], false, TOKENS_PER_CHUNK);
        file.add_chunk(vec![4, 5], true, 32);

        let (outer_bytes, payload) = file.to_bytes_uncompressed();

        let outer = CZIPv1OuterHeader::from_bytes(&outer_bytes).unwrap();
        let parsed = CompressedFile::read_from_decompressed(outer, &payload).unwrap();

        assert_eq!(parsed.inner_header.model_id_hash, 0xABCDEF00);
        assert_eq!(parsed.inner_header.chunk_count, 2);
        assert_eq!(parsed.inner_header.last_chunk_tokens, 32);
        assert_eq!(parsed.get_total_tokens(), 64 + 32);
    }

    #[test]
    fn test_file_roundtrip() {
        let mut file = CompressedFile::new(0x12345678, 1, Codec::Brotli, 2000);
        file.add_chunk(vec![1, 2, 3, 4, 5], false, TOKENS_PER_CHUNK);
        file.add_chunk(vec![6, 7, 8], false, TOKENS_PER_CHUNK);
        file.add_chunk(vec![9, 10], true, 48);

        let mut buf = Vec::new();
        file.write_to_uncompressed(&mut buf).unwrap();

        let mut cursor = Cursor::new(buf);
        let parsed = CompressedFile::read_from_uncompressed(&mut cursor).unwrap();

        assert_eq!(parsed.outer_header.codec_id, Codec::Brotli);
        assert_eq!(parsed.outer_header.uncompressed_len, 2000);
        assert_eq!(parsed.inner_header.model_id_hash, 0x12345678);
        assert_eq!(parsed.inner_header.tokenizer_id, 1);
        assert_eq!(parsed.inner_header.chunk_count, 3);
        assert_eq!(parsed.inner_header.last_chunk_tokens, 48);
        assert_eq!(parsed.chunks.len(), 3);
        assert_eq!(parsed.chunks[0].data, vec![1, 2, 3, 4, 5]);
        assert_eq!(parsed.chunks[1].data, vec![6, 7, 8]);
        assert_eq!(parsed.chunks[2].data, vec![9, 10]);
        assert_eq!(parsed.get_total_tokens(), 64 + 64 + 48);
        assert_eq!(parsed.get_chunk_tokens(0), 64);
        assert_eq!(parsed.get_chunk_tokens(1), 64);
        assert_eq!(parsed.get_chunk_tokens(2), 48);
    }
}
