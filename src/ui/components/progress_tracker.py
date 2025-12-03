"""
Progress Tracker Component

Provides multi-phase progress indicators and processing status displays.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from ..theme import ICONS, format_time


def processing_status(
    phases: List[str],
    current_phase: int,
    phase_progress: float,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display multi-phase processing indicator.
    
    Args:
        phases: List of phase names
        current_phase: Current phase index (0-based)
        phase_progress: Progress within current phase (0.0 to 1.0)
        details: Optional details dict with keys like 'files', 'speed', 'eta'
    """
    phase_icons = ["📄", "🧠", "💾", "✅"]
    
    # Phase indicator row
    cols = st.columns(len(phases))
    for i, phase in enumerate(phases):
        icon = phase_icons[i] if i < len(phase_icons) else "⏳"
        with cols[i]:
            if i < current_phase:
                st.markdown(f"✅ ~~{phase}~~")
            elif i == current_phase:
                st.markdown(f"**⏳ {phase}**")
            else:
                st.markdown(f"⬜ {phase}")
    
    # Progress bar
    st.progress(phase_progress)
    
    # Details row if provided
    if details:
        detail_cols = st.columns(3)
        with detail_cols[0]:
            if "files" in details:
                st.caption(f"📁 {details['files']}")
        with detail_cols[1]:
            if "speed" in details:
                st.caption(f"⚡ {details['speed']}")
        with detail_cols[2]:
            if "eta" in details:
                st.caption(f"⏱️ ETA: {details['eta']}")


def file_progress_tracker(
    total_files: int,
    processed_files: int,
    current_file: str = "",
    errors: int = 0
) -> None:
    """
    Display file processing progress.
    
    Args:
        total_files: Total number of files
        processed_files: Number of processed files
        current_file: Currently processing file name
        errors: Number of errors encountered
    """
    progress = processed_files / total_files if total_files > 0 else 0
    
    st.progress(progress, text=f"Processing file {processed_files}/{total_files}")
    
    if current_file:
        st.caption(f"📄 Current: {current_file}")
    
    if errors > 0:
        st.caption(f"⚠️ {errors} error(s) encountered")


def chunk_progress_tracker(
    total_chunks: int,
    processed_chunks: int,
    chunks_per_second: float = 0.0
) -> None:
    """
    Display chunk processing progress.
    
    Args:
        total_chunks: Total number of chunks
        processed_chunks: Number of processed chunks
        chunks_per_second: Processing speed
    """
    progress = processed_chunks / total_chunks if total_chunks > 0 else 0
    
    st.progress(progress)
    
    cols = st.columns(3)
    with cols[0]:
        st.caption(f"📝 Chunks: {processed_chunks}/{total_chunks}")
    with cols[1]:
        if chunks_per_second > 0:
            st.caption(f"⚡ {chunks_per_second:.1f} chunks/s")
    with cols[2]:
        if chunks_per_second > 0 and total_chunks > processed_chunks:
            remaining = total_chunks - processed_chunks
            eta_seconds = remaining / chunks_per_second
            st.caption(f"⏱️ ETA: {format_time(eta_seconds)}")


def embedding_progress_tracker(
    total_embeddings: int,
    generated_embeddings: int,
    batch_size: int = 0,
    current_batch: int = 0
) -> None:
    """
    Display embedding generation progress.
    
    Args:
        total_embeddings: Total embeddings to generate
        generated_embeddings: Generated embeddings count
        batch_size: Batch size being used
        current_batch: Current batch number
    """
    progress = generated_embeddings / total_embeddings if total_embeddings > 0 else 0
    
    st.progress(progress, text=f"Generating embeddings: {generated_embeddings}/{total_embeddings}")
    
    if batch_size > 0:
        total_batches = (total_embeddings + batch_size - 1) // batch_size
        st.caption(f"🔢 Batch {current_batch}/{total_batches} (size: {batch_size})")


def indexing_progress_tracker(
    total_docs: int,
    indexed_docs: int,
    skipped_docs: int = 0
) -> None:
    """
    Display indexing progress.
    
    Args:
        total_docs: Total documents to index
        indexed_docs: Indexed documents count
        skipped_docs: Skipped (duplicate) documents count
    """
    progress = (indexed_docs + skipped_docs) / total_docs if total_docs > 0 else 0
    
    st.progress(progress, text=f"Indexing: {indexed_docs}/{total_docs}")
    
    if skipped_docs > 0:
        st.caption(f"⏭️ {skipped_docs} duplicates skipped")


class ProgressContext:
    """
    Context manager for progress tracking with automatic cleanup.
    
    Usage:
        with ProgressContext("Processing files", total=10) as progress:
            for i, file in enumerate(files):
                process(file)
                progress.update(i + 1, f"Processing {file}")
    """
    
    def __init__(
        self,
        title: str,
        total: int = 100,
        show_spinner: bool = True
    ):
        """
        Initialize progress context.
        
        Args:
            title: Progress title
            total: Total steps
            show_spinner: Whether to show spinner
        """
        self.title = title
        self.total = total
        self.show_spinner = show_spinner
        self.current = 0
        self._progress_bar = None
        self._status_text = None
    
    def __enter__(self):
        if self.show_spinner:
            self._status = st.status(self.title, expanded=True)
            self._status.__enter__()
        self._progress_bar = st.progress(0)
        self._status_text = st.empty()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._progress_bar.progress(1.0)
            self._status_text.text("✅ Complete!")
        else:
            self._status_text.text(f"❌ Error: {exc_val}")
        
        if self.show_spinner:
            self._status.__exit__(exc_type, exc_val, exc_tb)
        
        return False
    
    def update(self, current: int, message: str = "") -> None:
        """
        Update progress.
        
        Args:
            current: Current step
            message: Status message
        """
        self.current = current
        progress = current / self.total if self.total > 0 else 0
        self._progress_bar.progress(progress)
        if message:
            self._status_text.text(message)
