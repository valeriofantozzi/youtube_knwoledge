"""
Load Documents Page Module

Implements the file upload and complete processing pipeline UI:
- File upload (single/multiple document files: SRT, TXT, MD)
- Model selection
- Processing options
- Progress visualization
- Processing statistics display
"""

import streamlit as st
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from ..state import (
    ProcessingStatus,
    ProcessingStats,
    update_processing_progress,
    add_processing_error,
    reset_processing_state,
)
from ..theme import ICONS, COLORS
from ..components.feedback import (
    show_success_toast,
    show_error_with_details,
    show_empty_state,
    show_info_callout,
    show_processing_errors,
    show_success_summary,
)
from ..components.progress_tracker import processing_status


# Supported file extensions
SUPPORTED_EXTENSIONS = {'.srt', '.sub', '.txt', '.text', '.md', '.markdown', '.mdown'}


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available embedding models from registry.
    
    Returns:
        List of model info dictionaries
    """
    try:
        from ...embeddings.model_registry import get_model_registry
        registry = get_model_registry()
        models = []
        for model_name in registry.get_registered_models():
            metadata = registry.get_model_metadata(model_name)
            if metadata:
                models.append({
                    "name": model_name,
                    "dimensions": metadata.embedding_dimension,
                    "max_seq_length": metadata.max_sequence_length,
                })
        return models
    except Exception as e:
        st.warning(f"Could not load model registry: {e}")
        # Return default model
        return [{
            "name": "BAAI/bge-large-en-v1.5",
            "dimensions": 1024,
            "max_seq_length": 512,
        }]


def validate_document_file(file) -> Tuple[bool, str]:
    """
    Validate an uploaded document file.
    
    Args:
        file: Streamlit uploaded file object
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    file_ext = Path(file.name).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        supported_str = ', '.join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"Invalid file type: {file.name}. Supported formats: {supported_str}"
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if file.size > max_size:
        return False, f"File too large: {file.name} ({file.size / 1024 / 1024:.1f}MB). Max size is 10MB."
    
    return True, ""


# Backward compatibility alias
validate_srt_file = validate_document_file


def save_uploaded_files(uploaded_files: List) -> Tuple[List[Path], List[str]]:
    """
    Save uploaded files to temporary directory.
    
    Args:
        uploaded_files: List of Streamlit uploaded file objects
    
    Returns:
        Tuple of (saved_paths, errors)
    """
    saved_paths = []
    errors = []
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="document_upload_"))
    
    for file in uploaded_files:
        try:
            # Validate file
            is_valid, error_msg = validate_document_file(file)
            if not is_valid:
                errors.append(error_msg)
                continue
            
            # Save to temp directory
            file_path = temp_dir / file.name
            file_path.write_bytes(file.getvalue())
            saved_paths.append(file_path)
        except Exception as e:
            errors.append(f"Error saving {file.name}: {str(e)}")
    
    return saved_paths, errors


def process_files(
    file_paths: List[Path],
    model_name: str,
    batch_size: int,
    skip_duplicates: bool,
    parallel_preprocessing: bool,
    progress_container
) -> Dict[str, Any]:
    """
    Process files through the complete pipeline.
    
    Args:
        file_paths: List of document file paths (SRT, TXT, MD, etc.)
        model_name: Embedding model name
        batch_size: Batch size for embedding generation
        skip_duplicates: Whether to skip already indexed documents
        parallel_preprocessing: Whether to use parallel preprocessing
        progress_container: Streamlit container for progress updates
    
    Returns:
        Processing results dictionary
    """
    from ...preprocessing.pipeline import PreprocessingPipeline
    from ...embeddings.pipeline import EmbeddingPipeline
    from ...vector_store.pipeline import VectorStorePipeline
    
    results = {
        "files_processed": 0,
        "files_total": len(file_paths),
        "chunks_created": 0,
        "embeddings_generated": 0,
        "documents_indexed": 0,
        "errors": [],
        "skipped": 0,
        "processing_time": 0.0,
    }
    
    start_time = time.time()
    
    try:
        # Initialize pipelines
        preprocessing_pipeline = PreprocessingPipeline()
        embedding_pipeline = EmbeddingPipeline(
            model_name=model_name,
            batch_size=batch_size,
            enable_optimizations=True
        )
        vector_store_pipeline = VectorStorePipeline(model_name=model_name)
        
        # Phase 1: Preprocessing
        with progress_container:
            st.write(f"{ICONS['processing']} **Phase 1/3: Preprocessing**")
            preprocess_progress = st.progress(0, text="Parsing document files...")
            
            processed_documents = []
            for i, file_path in enumerate(file_paths):
                try:
                    processed_document = preprocessing_pipeline.process_file(file_path)
                    if processed_document:
                        processed_documents.append(processed_document)
                        results["chunks_created"] += len(processed_document.chunks)
                    results["files_processed"] += 1
                except Exception as e:
                    results["errors"].append(f"Preprocessing error for {file_path.name}: {str(e)}")
                
                progress = (i + 1) / len(file_paths)
                preprocess_progress.progress(progress, text=f"Parsed {i + 1}/{len(file_paths)} files")
            
            preprocess_progress.progress(1.0, text="✅ Preprocessing complete")
        
        # Phase 2: Embedding Generation
        with progress_container:
            st.write(f"{ICONS['processing']} **Phase 2/3: Generating Embeddings**")
            embed_progress = st.progress(0, text="Generating embeddings...")
            speed_display = st.empty()
            
            all_embeddings = []
            total_chunks = results["chunks_created"]
            chunks_processed = 0
            
            for i, processed_document in enumerate(processed_documents):
                try:
                    start_embed = time.time()
                    embeddings, metadata, actual_model = embedding_pipeline.generate_embeddings(
                        processed_document,
                        show_progress=False
                    )
                    embed_time = time.time() - start_embed
                    
                    all_embeddings.append({
                        "document": processed_document,
                        "embeddings": embeddings,
                        "metadata": metadata,
                        "model_name": actual_model,
                    })
                    
                    chunks_processed += len(processed_document.chunks)
                    results["embeddings_generated"] += len(embeddings)
                    
                    # Update progress
                    progress = chunks_processed / total_chunks if total_chunks > 0 else 0
                    speed = len(processed_document.chunks) / embed_time if embed_time > 0 else 0
                    embed_progress.progress(
                        progress,
                        text=f"Generated embeddings for {i + 1}/{len(processed_documents)} documents"
                    )
                    speed_display.caption(f"⚡ {speed:.1f} chunks/sec")
                    
                except Exception as e:
                    results["errors"].append(
                        f"Embedding error for {processed_document.metadata.filename}: {str(e)}"
                    )
            
            embed_progress.progress(1.0, text="✅ Embeddings generated")
        
        # Phase 3: Indexing
        with progress_container:
            st.write(f"{ICONS['processing']} **Phase 3/3: Indexing**")
            index_progress = st.progress(0, text="Indexing in vector database...")
            
            for i, item in enumerate(all_embeddings):
                try:
                    indexed_count = vector_store_pipeline.index_processed_document(
                        processed_document=item["document"],
                        embeddings=item["embeddings"],
                        skip_duplicates=skip_duplicates,
                        show_progress=False,
                        model_name=item["model_name"]
                    )
                    results["documents_indexed"] += indexed_count
                    
                    if indexed_count == 0 and skip_duplicates:
                        results["skipped"] += len(item["document"].chunks)
                    
                except Exception as e:
                    results["errors"].append(
                        f"Indexing error for {item['document'].metadata.filename}: {str(e)}"
                    )
                
                progress = (i + 1) / len(all_embeddings)
                index_progress.progress(progress, text=f"Indexed {i + 1}/{len(all_embeddings)} documents")
            
            index_progress.progress(1.0, text="✅ Indexing complete")
    
    except Exception as e:
        results["errors"].append(f"Pipeline error: {str(e)}")
    
    results["processing_time"] = time.time() - start_time
    return results


def render_file_upload_section() -> List:
    """
    Render the file upload section.
    
    Returns:
        List of uploaded files
    """
    st.subheader(f"{ICONS['files']} Input Files")
    
    # Tabs for upload methods
    upload_tab, directory_tab = st.tabs(["📁 Upload Files", "📂 Directory Path"])
    
    uploaded_files = []
    
    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload document files (SRT, TXT, MD)",
            type=["srt", "sub", "txt", "text", "md", "markdown", "mdown"],
            accept_multiple_files=True,
            help="Drag and drop files here or click to browse. Supported: SRT, TXT, MD. Max 10MB per file."
        )
        
        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files)
            st.caption(
                f"{ICONS['document']} {len(uploaded_files)} file(s) selected "
                f"({total_size / 1024:.1f} KB total)"
            )
            
            # Show file list in expander
            with st.expander("View selected files", expanded=False):
                for f in uploaded_files:
                    st.text(f"📄 {f.name} ({f.size / 1024:.1f} KB)")
    
    with directory_tab:
        dir_path = st.text_input(
            "Directory path containing document files",
            placeholder="/path/to/documents/",
            help="Enter the absolute path to a directory containing document files."
        )
        
        if dir_path:
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                # Find all supported files
                doc_files = []
                for ext in SUPPORTED_EXTENSIONS:
                    doc_files.extend(path.glob(f"*{ext}"))
                    doc_files.extend(path.glob(f"*{ext.upper()}"))
                doc_files = sorted(set(doc_files))
                
                if doc_files:
                    st.success(f"✅ Found {len(doc_files)} document files")
                    with st.expander("View files", expanded=False):
                        for f in doc_files[:20]:  # Show first 20
                            st.text(f"📄 {f.name}")
                        if len(doc_files) > 20:
                            st.caption(f"... and {len(doc_files) - 20} more")
                    
                    # Store directory files in session for processing
                    st.session_state["directory_files"] = doc_files
                else:
                    st.warning("⚠️ No supported document files found in this directory")
            elif dir_path:
                st.error("❌ Directory not found")
    
    return uploaded_files


def render_model_selection_section() -> str:
    """
    Render the model selection section.
    
    Returns:
        Selected model name
    """
    st.subheader(f"{ICONS['robot']} Model Selection")
    
    models = get_available_models()
    model_names = [m["name"] for m in models]
    
    # Get default from config or use first model
    try:
        from ...utils.config import get_config
        config = get_config()
        default_model = getattr(config, "MODEL_NAME", model_names[0])
    except Exception:
        default_model = model_names[0]
    
    default_index = model_names.index(default_model) if default_model in model_names else 0
    
    selected_model = st.selectbox(
        "Embedding Model",
        options=model_names,
        index=default_index,
        help="Select the model to use for generating embeddings."
    )
    
    # Show model info
    model_info = next((m for m in models if m["name"] == selected_model), None)
    if model_info:
        st.caption(
            f"ℹ️ {model_info['dimensions']} dimensions • "
            f"Max {model_info['max_seq_length']} tokens"
        )
    
    return selected_model


def render_processing_options_section() -> Dict[str, Any]:
    """
    Render the processing options section.
    
    Returns:
        Dictionary of processing options
    """
    with st.expander(f"{ICONS['settings']} Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.slider(
                "Batch Size",
                min_value=16,
                max_value=256,
                value=64,
                step=16,
                help="Number of chunks to process at once. Higher values use more memory but are faster."
            )
            
            skip_duplicates = st.checkbox(
                "Skip already indexed documents",
                value=True,
                help="Skip documents that have already been indexed to avoid duplicates."
            )
        
        with col2:
            parallel_preprocessing = st.checkbox(
                "Enable parallel preprocessing",
                value=True,
                help="Use multiple processes for parsing SRT files."
            )
    
    return {
        "batch_size": batch_size,
        "skip_duplicates": skip_duplicates,
        "parallel_preprocessing": parallel_preprocessing,
    }


def render_load_documents_page() -> None:
    """
    Render the Load Documents page.
    
    This is the main entry point for the page module.
    """
    st.header(f"{ICONS['load']} Load Documents")
    st.markdown("Import subtitle files and generate embeddings for semantic search.")
    
    st.markdown("---")
    
    # File upload section
    uploaded_files = render_file_upload_section()
    
    st.markdown("---")
    
    # Model selection
    selected_model = render_model_selection_section()
    
    # Processing options
    options = render_processing_options_section()
    
    st.markdown("---")
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            f"{ICONS['run']} Start Processing",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files and "directory_files" not in st.session_state
        )
    
    # Processing section
    if process_button:
        # Determine files to process
        files_to_process = []
        
        if uploaded_files:
            # Save uploaded files and get paths
            saved_paths, save_errors = save_uploaded_files(uploaded_files)
            files_to_process = saved_paths
            
            if save_errors:
                for error in save_errors:
                    st.warning(error)
        elif "directory_files" in st.session_state:
            files_to_process = st.session_state["directory_files"]
        
        if not files_to_process:
            st.error("❌ No valid files to process")
            return
        
        st.markdown("---")
        st.subheader(f"{ICONS['processing']} Processing Progress")
        
        # Create progress container
        progress_container = st.container()
        
        # Process files
        with st.spinner("Processing..."):
            results = process_files(
                file_paths=files_to_process,
                model_name=selected_model,
                batch_size=options["batch_size"],
                skip_duplicates=options["skip_duplicates"],
                parallel_preprocessing=options["parallel_preprocessing"],
                progress_container=progress_container
            )
        
        # Display results
        st.markdown("---")
        st.subheader(f"{ICONS['chart']} Processing Results")
        
        if results["errors"]:
            show_processing_errors(results["errors"])
        
        # Success summary
        if results["documents_indexed"] > 0 or results["files_processed"] > 0:
            show_success_summary(
                title="Processing Complete",
                stats={
                    "Files Processed": results["files_processed"],
                    "Chunks Created": results["chunks_created"],
                    "Documents Indexed": results["documents_indexed"],
                    "Processing Time": f"{results['processing_time']:.1f}s",
                }
            )
            
            if results["skipped"] > 0:
                st.info(f"ℹ️ {results['skipped']} duplicate chunks were skipped")
            
            # Refresh collection count
            try:
                from ...vector_store.chroma_manager import ChromaDBManager
                manager = ChromaDBManager()
                st.session_state.collection = manager.get_or_create_collection()
                st.session_state.total_docs = st.session_state.collection.count()
            except Exception:
                pass
            
            # Navigation button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    f"{ICONS['analysis']} Go to PostProcessing",
                    use_container_width=True
                ):
                    st.session_state.current_page = "🔬 PostProcessing"
                    st.rerun()
        else:
            st.warning("⚠️ No documents were indexed. Check the errors above.")
