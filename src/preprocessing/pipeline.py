"""
Preprocessing Pipeline Module

Orchestrates the complete preprocessing pipeline: document parsing, text cleaning,
chunking, and metadata attachment. Supports multiple file formats through pluggable parsers.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

from .srt_parser import SRTParser, SubtitleEntry
from .text_cleaner import TextCleaner
from .chunker import SemanticChunker, Chunk
from .metadata_extractor import MetadataExtractor
from .parser_base import (
    DocumentParser, 
    ParserRegistry, 
    TextEntry, 
    SourceMetadata,
    find_document_files
)
from ..utils.logger import get_default_logger
from ..utils.config import get_config


@dataclass
class ProcessedDocument:
    """
    Represents a fully processed document.
    
    This is the generalized version that works with any document type.
    """
    metadata: SourceMetadata
    entries: Union[List[TextEntry], List[SubtitleEntry]]
    chunks: List[Chunk]
    stats: Dict
    content_type: str = "unknown"  # File type: "srt", "txt", "md", etc.
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "stats": self.stats,
            "content_type": self.content_type,
        }
    
    @property
    def source_id(self) -> str:
        """Get source ID from metadata."""
        return self.metadata.source_id


class PreprocessingPipeline:
    """
    Main preprocessing pipeline orchestrator.
    
    Supports multiple document types through the ParserRegistry.
    Falls back to SRT parser for backward compatibility.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            chunk_size: Chunk size in tokens (default from config)
            chunk_overlap: Overlap size in tokens (default from config)
            min_chunk_size: Minimum chunk size in tokens (default from config)
        """
        self.parser = SRTParser()  # Legacy parser for backward compatibility
        self.cleaner = TextCleaner()
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        self.metadata_extractor = MetadataExtractor()
        self.logger = get_default_logger()
        self.config = get_config()
    
    def process_file(
        self,
        file_path: Path,
        skip_errors: bool = True
    ) -> Optional[ProcessedDocument]:
        """
        Process a single document file through the complete pipeline.
        
        Automatically selects the appropriate parser based on file extension.
        
        Args:
            file_path: Path to document file
            skip_errors: If True, return None on errors instead of raising
        
        Returns:
            ProcessedDocument object or None if error and skip_errors=True
        """
        try:
            # Get appropriate parser from registry
            parser = ParserRegistry.get_parser(file_path)
            
            if parser:
                # Use new parser infrastructure
                return self._process_with_parser(file_path, parser)
            else:
                # Fall back to legacy SRT processing for .srt files
                if file_path.suffix.lower() in ['.srt', '.sub']:
                    return self._process_srt_legacy(file_path)
                else:
                    self.logger.warning(
                        f"No parser found for {file_path.suffix}. "
                        f"Supported formats: {ParserRegistry.supported_extensions()}"
                    )
                    return None
        
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
            if skip_errors:
                return None
            raise
    
    def _process_with_parser(
        self,
        file_path: Path,
        parser: DocumentParser
    ) -> Optional[ProcessedDocument]:
        """
        Process file using the new parser infrastructure.
        
        Args:
            file_path: Path to document file
            parser: DocumentParser instance
            
        Returns:
            ProcessedDocument object or None
        """
        self.logger.debug(f"Processing {file_path.name} with {parser.parser_name}")
        
        # Parse document
        entries, metadata = parser.parse(file_path)
        
        if not entries:
            self.logger.warning(f"No content entries found in {file_path.name}")
            return None
        
        # Extract text from entries
        texts = [entry.text for entry in entries]
        all_text = ' '.join(texts)
        
        # Clean text
        cleaned_text = self.cleaner.clean_text(all_text)
        
        # Create chunks with metadata
        chunk_metadata = {
            "source_id": metadata.source_id,
            "date": metadata.date,
            "title": metadata.title,
            "filename": metadata.original_filename,
            "source_type": metadata.source_type,
        }
        chunks = self.chunker.chunk_text(cleaned_text, chunk_metadata)
        
        # Calculate statistics
        stats = self._calculate_stats_generic(entries, chunks, cleaned_text)
        
        self.logger.info(
            f"Processed {file_path.name}: {len(entries)} entries, "
            f"{len(chunks)} chunks, {stats['total_tokens']} tokens"
        )
        
        return ProcessedDocument(
            metadata=metadata,
            entries=entries,
            chunks=chunks,
            stats=stats,
            content_type=metadata.source_type
        )
    
    def _process_srt_legacy(self, file_path: Path) -> Optional[ProcessedDocument]:
        """
        Legacy SRT processing for backward compatibility.
        
        Args:
            file_path: Path to SRT file
            
        Returns:
            ProcessedDocument object or None
        """
        # Extract metadata using legacy extractor
        self.logger.debug(f"Extracting metadata from {file_path.name}")
        try:
            metadata = self.metadata_extractor.extract_from_filename(file_path)
        except ValueError as e:
            self.logger.warning(
                f"Could not extract metadata from {file_path.name}: {e}. "
                "Using default metadata."
            )
            metadata = SourceMetadata(
                source_id="unknown",
                date="0000/00/00",
                title=file_path.stem,
                source_type="srt",
                original_filename=file_path.name,
                file_path=str(file_path.absolute())
            )
        
        # Parse SRT file
        self.logger.debug(f"Parsing SRT file: {file_path.name}")
        entries = self.parser.parse_file(file_path)
        
        if not entries:
            self.logger.warning(f"No subtitle entries found in {file_path.name}")
            return None
        
        # Clean text from entries
        self.logger.debug(f"Cleaning text from {len(entries)} entries")
        cleaned_texts = self.cleaner.clean_subtitle_entries(entries)
        all_cleaned_text = ' '.join(cleaned_texts)
        
        # Create chunks
        self.logger.debug(f"Creating chunks from cleaned text")
        chunk_metadata = {
            "source_id": metadata.source_id,
            "date": metadata.date,
            "title": metadata.title,
            "filename": metadata.original_filename,
            "source_type": "srt",
        }
        chunks = self.chunker.chunk_text(all_cleaned_text, chunk_metadata)
        
        # Calculate statistics
        stats = self._calculate_stats(entries, chunks, all_cleaned_text)
        
        self.logger.info(
            f"Processed {file_path.name}: {len(entries)} entries, "
            f"{len(chunks)} chunks, {stats['total_tokens']} tokens"
        )
        
        return ProcessedDocument(
            metadata=metadata,
            entries=entries,
            chunks=chunks,
            stats=stats,
            content_type="srt"
        )
    
    def process_multiple_files(
        self,
        file_paths: List[Path],
        skip_errors: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> List[ProcessedDocument]:
        """
        Process multiple document files, optionally in parallel.
        
        Args:
            file_paths: List of paths to document files
            skip_errors: If True, skip files that fail to process
            parallel: If True, use parallel processing
            max_workers: Maximum number of worker processes/threads (default from config)
        
        Returns:
            List of ProcessedDocument objects
        """
        if not file_paths:
            return []
        
        if not parallel or len(file_paths) == 1:
            # Sequential processing
            results = []
            for file_path in file_paths:
                processed = self.process_file(file_path, skip_errors=skip_errors)
                if processed:
                    results.append(processed)
            return results
        
        # Parallel processing
        max_workers = max_workers or self.config.MAX_WORKERS
        self.logger.info(f"Processing {len(file_paths)} files in parallel with {max_workers} workers")
        
        # Use ThreadPoolExecutor for I/O-bound operations (file reading)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_file, file_path, skip_errors): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    processed = future.result()
                    if processed:
                        results.append(processed)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}", exc_info=True)
                    if not skip_errors:
                        raise
        
        self.logger.info(f"Successfully processed {len(results)}/{len(file_paths)} files")
        return results
    
    def _calculate_stats(
        self,
        entries: Union[List[SubtitleEntry], List[TextEntry]],
        chunks: List[Chunk],
        text: str
    ) -> Dict:
        """
        Calculate processing statistics.
        
        Args:
            entries: List of text entries (SubtitleEntry or TextEntry)
            chunks: List of chunks
            text: Full cleaned text
        
        Returns:
            Statistics dictionary
        """
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_chunk_size = total_tokens / len(chunks) if chunks else 0
        
        return {
            "total_entries": len(entries),
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_characters": len(text),
            "avg_chunk_size": round(avg_chunk_size, 2),
            "min_chunk_size": min((c.token_count for c in chunks), default=0),
            "max_chunk_size": max((c.token_count for c in chunks), default=0),
        }
    
    def _calculate_stats_generic(
        self,
        entries: List[TextEntry],
        chunks: List[Chunk],
        text: str
    ) -> Dict:
        """
        Calculate processing statistics for generic text entries.
        
        Args:
            entries: List of TextEntry objects
            chunks: List of chunks
            text: Full cleaned text
        
        Returns:
            Statistics dictionary
        """
        return self._calculate_stats(entries, chunks, text)
    
    def save_processed_chunks(
        self,
        processed_document: ProcessedDocument,
        output_dir: Path,
        format: str = "json"
    ) -> Path:
        """
        Save processed chunks to disk.
        
        Args:
            processed_document: ProcessedDocument object
            output_dir: Output directory
            format: Output format ("json" or "parquet")
        
        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use source_id for filename
        source_id = processed_document.source_id
        
        if format == "json":
            output_file = output_dir / f"{source_id}_chunks.json"
            data = processed_document.to_dict()
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_file
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        skip_errors: bool = True
    ) -> Tuple[List[ProcessedDocument], Dict]:
        """
        Process all document files in a directory.
        
        Args:
            input_dir: Input directory containing document files
            output_dir: Optional output directory for saved chunks
            pattern: File pattern to match (deprecated, use extensions)
            extensions: List of file extensions to process (default: all supported)
            skip_errors: If True, skip files that fail to process
        
        Returns:
            Tuple of (list of ProcessedDocument objects, summary statistics)
        """
        # Handle backward compatibility with pattern argument
        if pattern and not extensions:
            # Extract extension from pattern like "*.srt"
            if pattern.startswith("*."):
                extensions = [pattern[1:]]  # Remove "*" 
            else:
                # Fall back to old behavior
                doc_files = list(input_dir.rglob(pattern))
                self.logger.info(f"Found {len(doc_files)} files matching {pattern} in {input_dir}")
                processed_docs = self.process_multiple_files(doc_files, skip_errors=skip_errors)
                summary = self._calculate_summary_stats(processed_docs)
                return processed_docs, summary
        
        # Find all document files using new infrastructure
        doc_files = find_document_files(input_dir, extensions=extensions, recursive=True)
        self.logger.info(f"Found {len(doc_files)} document files in {input_dir}")
        
        # Process files
        processed_docs = self.process_multiple_files(doc_files, skip_errors=skip_errors)
        
        # Save chunks if output directory specified
        if output_dir:
            self.logger.info(f"Saving processed chunks to {output_dir}")
            for processed in processed_docs:
                try:
                    self.save_processed_chunks(processed, output_dir)
                except Exception as e:
                    self.logger.error(f"Error saving chunks: {e}")
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(processed_docs)
        
        return processed_docs, summary
    
    def _calculate_summary_stats(
        self,
        processed_documents: List[ProcessedDocument]
    ) -> Dict:
        """
        Calculate summary statistics across all processed documents.
        
        Args:
            processed_documents: List of ProcessedDocument objects
        
        Returns:
            Summary statistics dictionary
        """
        if not processed_documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_tokens": 0,
            }
        
        total_chunks = sum(len(pd.chunks) for pd in processed_documents)
        total_tokens = sum(pd.stats["total_tokens"] for pd in processed_documents)
        total_entries = sum(pd.stats["total_entries"] for pd in processed_documents)
        
        # Count by content type
        content_types = {}
        for pd in processed_documents:
            ct = pd.content_type
            content_types[ct] = content_types.get(ct, 0) + 1
        
        return {
            "total_documents": len(processed_documents),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "total_entries": total_entries,
            "avg_chunks_per_document": round(total_chunks / len(processed_documents), 2),
            "avg_tokens_per_document": round(total_tokens / len(processed_documents), 2),
            "content_types": content_types,
        }


# Backward compatibility aliases for function names
process_srt_files = PreprocessingPipeline.process_multiple_files