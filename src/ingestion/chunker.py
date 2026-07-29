import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import tiktoken

from src.ingestion.loader import Document

logger = logging.getLogger(__name__)

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(TOKENIZER.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncates text to a maximum number of tokens."""
    tokens = TOKENIZER.encode(text)
    return TOKENIZER.decode(tokens[:max_tokens])


@dataclass
class Chunk:
    """Represents a single chunk of text ready for indexing."""

    text: str
    token_count: int
    chunk_index: int
    strategy: str
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.effective_chunk_size = chunk_size - overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Splits a document into chunks."""

    def _build_chunk(self, text: str, index: int, metadata: dict) -> Chunk:
        """Creates a Chunk object from text and metadata."""
        text = text.strip()
        return Chunk(
            text=text,
            token_count=count_tokens(text),
            chunk_index=index,
            strategy=self.__class__.__name__,
            metadata=metadata,
        )

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Applies overlap between consecutive chunks.
        The end of each chunk is prepended to the beginning of the next.
        """
        if self.overlap == 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            previous_tokens = TOKENIZER.encode(chunks[i - 1])
            overlap_tokens = previous_tokens[-self.overlap :]
            overlap_text = TOKENIZER.decode(overlap_tokens)
            result.append(overlap_text + " " + chunks[i])

        return result


class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size chunks by token count.
    Simplest strategy — used as a baseline for comparison.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        logger.info(
            f"[FixedSize] Chunking {document.metadata['source']} "
            f"(size={self.chunk_size}, overlap={self.overlap})"
        )

        tokens = TOKENIZER.encode(document.content)
        step = self.effective_chunk_size - self.overlap
        raw_chunks = []

        for start in range(0, len(tokens), step):
            end = start + self.effective_chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = TOKENIZER.decode(chunk_tokens)
            raw_chunks.append(chunk_text)

            if end >= len(tokens):
                break

        chunks = []
        for i, text in enumerate(raw_chunks):
            chunk = self._build_chunk(
                text=text, index=i, metadata={**document.metadata, "chunk_index": i}
            )
            chunks.append(chunk)

        logger.info(f"[FixedSize] Generated {len(chunks)} chunks")
        return chunks


class SentenceChunker(BaseChunker):
    """
    Splits text at sentence boundaries.
    Chunks always end at the end of a complete sentence.
    """

    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    def _split_sentences(self, text: str) -> list[str]:
        """Splits text into individual sentences."""
        sentences = self.SENTENCE_ENDINGS.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, document: Document) -> list[Chunk]:
        logger.info(
            f"[Sentence] Chunking {document.metadata['source']} "
            f"(size={self.chunk_size}, overlap={self.overlap})"
        )

        sentences = self._split_sentences(document.content)
        raw_chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            # if a single sentence exceeds chunk_size, truncate it
            if sentence_tokens > self.effective_chunk_size:
                if current_chunk:
                    raw_chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                raw_chunks.append(
                    truncate_to_tokens(sentence, self.effective_chunk_size)
                )
                continue

            if current_tokens + sentence_tokens > self.effective_chunk_size:
                raw_chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            raw_chunks.append(" ".join(current_chunk))

        overlapped = self._apply_overlap(raw_chunks)

        chunks = []
        for i, text in enumerate(overlapped):
            chunk = self._build_chunk(
                text=text, index=i, metadata={**document.metadata, "chunk_index": i}
            )
            chunks.append(chunk)

        logger.info(f"[Sentence] Generated {len(chunks)} chunks")
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Splits text using a hierarchy of separators.
    Tries paragraphs first, then sentences, then fixed size as last resort.
    """

    SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", " "]

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Splits text by a separator, keeping non-empty parts."""
        parts = text.split(separator)
        return [p.strip() for p in parts if p.strip()]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively splits text using the separator hierarchy.
        Falls back to the next separator when chunks are still too large.
        """
        if count_tokens(text) <= self.effective_chunk_size:
            return [text]

        if not separators:
            return [truncate_to_tokens(text, self.effective_chunk_size)]

        separator = separators[0]
        remaining_separators = separators[1:]
        parts = self._split_by_separator(text, separator)

        result = []
        current_chunk = []
        current_tokens = 0

        for part in parts:
            part_tokens = count_tokens(part)

            if part_tokens > self.effective_chunk_size:
                # part is too large — recurse with next separator
                if current_chunk:
                    result.append(separator.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                sub_chunks = self._recursive_split(part, remaining_separators)
                result.extend(sub_chunks)
                continue

            if current_tokens + part_tokens > self.effective_chunk_size:
                result.append(separator.join(current_chunk))
                current_chunk = [part]
                current_tokens = part_tokens
            else:
                current_chunk.append(part)
                current_tokens += part_tokens

        if current_chunk:
            result.append(separator.join(current_chunk))

        return result

    def chunk(self, document: Document) -> list[Chunk]:
        logger.info(
            f"[Recursive] Chunking {document.metadata['source']} "
            f"(size={self.chunk_size}, overlap={self.overlap})"
        )

        raw_chunks = self._recursive_split(document.content, self.SEPARATORS)
        overlapped = self._apply_overlap(raw_chunks)

        chunks = []
        for i, text in enumerate(overlapped):
            chunk = self._build_chunk(
                text=text, index=i, metadata={**document.metadata, "chunk_index": i}
            )
            chunks.append(chunk)

        logger.info(f"[Recursive] Generated {len(chunks)} chunks")
        return chunks
