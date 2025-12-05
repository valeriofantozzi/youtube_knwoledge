"""
Chunker Module

Creates semantic chunks from subtitle text with overlap.
"""

import re
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.utils.config import get_config


@dataclass
class Chunk:
    """Represents a text chunk."""
    chunk_id: str
    text: str
    chunk_index: int
    token_count: int
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


class SemanticChunker:
    """Creates semantic chunks from text."""
    
    # Sentence ending patterns
    SENTENCE_END_PATTERN = re.compile(r'[.!?]+\s+')
    
    # Paragraph break patterns
    PARAGRAPH_BREAK_PATTERN = re.compile(r'\n\s*\n+')
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (default from config)
            chunk_overlap: Overlap size in tokens (default from config)
            min_chunk_size: Minimum chunk size in tokens (default from config)
        """
        config = get_config()
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or config.MIN_CHUNK_SIZE
        
        # Validate configuration
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_size > self.chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) must be <= chunk_size ({self.chunk_size})"
            )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk text into semantic segments.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
        
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Group sentences into chunks
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if (current_token_count + sentence_tokens > self.chunk_size and 
                current_chunk_sentences):
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                if current_token_count >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        chunk_text,
                        chunk_index,
                        current_token_count,
                        metadata,
                        source_id=metadata.get('source_id')
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences,
                    self.chunk_overlap
                )
                current_chunk_sentences = overlap_sentences
                current_token_count = sum(
                    self._count_tokens(s) for s in overlap_sentences
                )
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
        
        # Add final chunk if it meets minimum size
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            if current_token_count >= self.min_chunk_size:
                chunk = self._create_chunk(
                    chunk_text,
                    chunk_index,
                    current_token_count,
                    metadata,
                    source_id=metadata.get('source_id')
                )
                chunks.append(chunk)
            elif chunks:
                # Merge small final chunk with previous chunk
                chunks[-1].text += ' ' + chunk_text
                chunks[-1].token_count += current_token_count
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        # First, split by paragraph breaks
        paragraphs = self.PARAGRAPH_BREAK_PATTERN.split(text)
        
        sentences = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Split paragraph into sentences
            para_sentences = self.SENTENCE_END_PATTERN.split(paragraph)
            
            # Re-add sentence endings (they were removed by split)
            for i, sentence in enumerate(para_sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Add period if sentence doesn't end with punctuation
                if sentence and not sentence[-1] in '.!?':
                    sentence += '.'
                
                sentences.append(sentence)
        
        # If no sentence endings found, treat entire text as one sentence
        if not sentences:
            sentences = [text.strip()]
        
        return sentences
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text (simple word-based approximation).
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Approximate token count
        """
        # Simple approximation: split by whitespace and punctuation
        # This is a rough estimate; for exact tokenization, use a tokenizer
        words = re.findall(r'\b\w+\b', text)
        return len(words)
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        overlap_tokens: int
    ) -> List[str]:
        """
        Get sentences for overlap from end of chunk.
        
        Args:
            sentences: List of sentences
            overlap_tokens: Target overlap in tokens
        
        Returns:
            List of sentences that provide approximately overlap_tokens tokens
        """
        if not sentences:
            return []
        
        overlap_sentences = []
        token_count = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence)
            if token_count + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                token_count += sentence_tokens
            else:
                break
        
        return overlap_sentences if overlap_sentences else [sentences[-1]]
    
    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        token_count: int,
        metadata: Dict,
        source_id: Optional[str] = None
    ) -> Chunk:
        """
        Create a Chunk object with deterministic ID.
        
        Args:
            text: Chunk text
            chunk_index: Index of chunk in sequence
            token_count: Number of tokens in chunk
            metadata: Metadata dictionary
            source_id: Source document ID (used for deterministic chunk_id)
        
        Returns:
            Chunk object
        """
        # Generate deterministic chunk_id from source_id + chunk_index
        # This ensures same chunk from same document always gets same ID
        if source_id:
            # Combine source_id and chunk_index for deterministic ID
            deterministic_input = f"{source_id}_{chunk_index}".encode('utf-8')
            chunk_id = hashlib.sha256(deterministic_input).hexdigest()[:16]
        else:
            # Fallback: create ID from text hash + chunk_index
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
            chunk_id = f"{text_hash}_{chunk_index}"
        
        # Calculate chunk content hash for cross-file deduplication
        chunk_content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # Create metadata copy and update content_hash
        chunk_metadata = metadata.copy()
        chunk_metadata['content_hash'] = chunk_content_hash

        return Chunk(
            chunk_id=chunk_id,
            text=text.strip(),
            chunk_index=chunk_index,
            token_count=token_count,
            metadata=chunk_metadata
        )
    
    def chunk_subtitle_entries(
        self,
        entries: List,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk text from subtitle entries.
        
        Args:
            entries: List of subtitle entry objects with 'text' attribute
            metadata: Metadata to attach to chunks
        
        Returns:
            List of Chunk objects
        """
        # Combine all entry texts
        text = ' '.join(entry.text for entry in entries)
        
        return self.chunk_text(text, metadata)
    
    def chunk_multiple_texts(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[List[Chunk]]:
        """
        Chunk multiple texts.
        
        Args:
            texts: List of texts to chunk
            metadata_list: Optional list of metadata dicts (one per text)
        
        Returns:
            List of chunk lists (one per input text)
        """
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        return [
            self.chunk_text(text, metadata)
            for text, metadata in zip(texts, metadata_list)
        ]
