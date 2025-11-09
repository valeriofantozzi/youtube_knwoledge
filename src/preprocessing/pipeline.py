"""
Preprocessing Pipeline Module

Orchestrates the complete preprocessing pipeline: SRT parsing, text cleaning,
chunking, and metadata attachment.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

from .srt_parser import SRTParser, SubtitleEntry
from .text_cleaner import TextCleaner
from .chunker import SemanticChunker, Chunk
from .metadata_extractor import MetadataExtractor, VideoMetadata
from ..utils.logger import get_default_logger
from ..utils.config import get_config


@dataclass
class ProcessedVideo:
    """Represents a fully processed video."""
    metadata: VideoMetadata
    entries: List[SubtitleEntry]
    chunks: List[Chunk]
    stats: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "stats": self.stats,
        }


class PreprocessingPipeline:
    """Main preprocessing pipeline orchestrator."""
    
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
        self.parser = SRTParser()
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
    ) -> Optional[ProcessedVideo]:
        """
        Process a single SRT file through the complete pipeline.
        
        Args:
            file_path: Path to SRT file
            skip_errors: If True, return None on errors instead of raising
        
        Returns:
            ProcessedVideo object or None if error and skip_errors=True
        """
        try:
            # Extract metadata
            self.logger.debug(f"Extracting metadata from {file_path.name}")
            try:
                metadata = self.metadata_extractor.extract_from_filename(file_path)
            except ValueError as e:
                # If metadata extraction fails, create default metadata
                self.logger.warning(
                    f"Could not extract metadata from {file_path.name}: {e}. "
                    "Using default metadata."
                )
                from .metadata_extractor import VideoMetadata
                metadata = VideoMetadata(
                    video_id="unknown",
                    date="0000/00/00",
                    title=file_path.stem,
                    filename=file_path.name,
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
                "video_id": metadata.video_id,
                "date": metadata.date,
                "title": metadata.title,
                "filename": metadata.filename,
            }
            chunks = self.chunker.chunk_text(all_cleaned_text, chunk_metadata)
            
            # Calculate statistics
            stats = self._calculate_stats(entries, chunks, all_cleaned_text)
            
            self.logger.info(
                f"Processed {file_path.name}: {len(entries)} entries, "
                f"{len(chunks)} chunks, {stats['total_tokens']} tokens"
            )
            
            return ProcessedVideo(
                metadata=metadata,
                entries=entries,
                chunks=chunks,
                stats=stats
            )
        
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
            if skip_errors:
                return None
            raise
    
    def process_multiple_files(
        self,
        file_paths: List[Path],
        skip_errors: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> List[ProcessedVideo]:
        """
        Process multiple SRT files, optionally in parallel.
        
        Args:
            file_paths: List of paths to SRT files
            skip_errors: If True, skip files that fail to process
            parallel: If True, use parallel processing
            max_workers: Maximum number of worker processes/threads (default from config)
        
        Returns:
            List of ProcessedVideo objects
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
        # ProcessPoolExecutor would require pickling the entire pipeline, which is complex
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
        entries: List[SubtitleEntry],
        chunks: List[Chunk],
        text: str
    ) -> Dict:
        """
        Calculate processing statistics.
        
        Args:
            entries: List of subtitle entries
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
    
    def save_processed_chunks(
        self,
        processed_video: ProcessedVideo,
        output_dir: Path,
        format: str = "json"
    ) -> Path:
        """
        Save processed chunks to disk.
        
        Args:
            processed_video: ProcessedVideo object
            output_dir: Output directory
            format: Output format ("json" or "parquet")
        
        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            output_file = output_dir / f"{processed_video.metadata.video_id}_chunks.json"
            data = processed_video.to_dict()
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_file
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: str = "*.srt",
        skip_errors: bool = True
    ) -> Tuple[List[ProcessedVideo], Dict]:
        """
        Process all SRT files in a directory.
        
        Args:
            input_dir: Input directory containing SRT files
            output_dir: Optional output directory for saved chunks
            pattern: File pattern to match
            skip_errors: If True, skip files that fail to process
        
        Returns:
            Tuple of (list of ProcessedVideo objects, summary statistics)
        """
        # Find all SRT files
        srt_files = list(input_dir.rglob(pattern))
        self.logger.info(f"Found {len(srt_files)} SRT files in {input_dir}")
        
        # Process files
        processed_videos = self.process_multiple_files(srt_files, skip_errors=skip_errors)
        
        # Save chunks if output directory specified
        if output_dir:
            self.logger.info(f"Saving processed chunks to {output_dir}")
            for processed in processed_videos:
                try:
                    self.save_processed_chunks(processed, output_dir)
                except Exception as e:
                    self.logger.error(f"Error saving chunks: {e}")
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(processed_videos)
        
        return processed_videos, summary
    
    def _calculate_summary_stats(
        self,
        processed_videos: List[ProcessedVideo]
    ) -> Dict:
        """
        Calculate summary statistics across all processed videos.
        
        Args:
            processed_videos: List of ProcessedVideo objects
        
        Returns:
            Summary statistics dictionary
        """
        if not processed_videos:
            return {
                "total_videos": 0,
                "total_chunks": 0,
                "total_tokens": 0,
            }
        
        total_chunks = sum(len(pv.chunks) for pv in processed_videos)
        total_tokens = sum(pv.stats["total_tokens"] for pv in processed_videos)
        total_entries = sum(pv.stats["total_entries"] for pv in processed_videos)
        
        return {
            "total_videos": len(processed_videos),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "total_entries": total_entries,
            "avg_chunks_per_video": round(total_chunks / len(processed_videos), 2),
            "avg_tokens_per_video": round(total_tokens / len(processed_videos), 2),
        }

