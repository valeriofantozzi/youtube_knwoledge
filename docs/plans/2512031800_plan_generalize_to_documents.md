# Implementation Plan: Generalize Project from Video/SRT to Generic Documents

**Plan ID:** `2512031800_plan_generalize_to_documents`  
**Created:** 2025-12-03  
**Status:** Draft  
**Last Updated:** 2025-12-03

---

## Technical & Engineering Description

### Overview

**What is being built:**  
Complete refactoring of the project to support any document type instead of being limited to video subtitles (SRT files). The system will become a generic **Document Embedding & Retrieval Platform** capable of:

1. **Loading documents** of various types (SRT, TXT, MD, PDF, etc.)
2. **Processing and chunking** content with format-specific parsers
3. **Generating embeddings** (unchanged core functionality)
4. **Searching and clustering** documents semantically

**Business/Functional Goals:**

- Transform from a YouTube-specific tool to a **generic knowledge base system**
- Support multiple document formats through pluggable parsers
- Maintain backward compatibility with existing SRT data
- Enable users to build knowledge bases from any text-based content

---

### Architecture Changes

**Current Architecture (Video-Specific):**

```
SRT Files → SRTParser → ProcessedVideo → Embeddings → ChromaDB
                          (video_id)                  (video_id metadata)
```

**Target Architecture (Generic Documents):**

```
Any Document → ParserRegistry → ProcessedDocument → Embeddings → ChromaDB
   │              │                 (source_id)                  (source_id metadata)
   │              │
   │              ├── SRTParser (for .srt files)
   │              ├── TextParser (for .txt files)
   │              ├── MarkdownParser (for .md files)
   │              └── [Future: PDFParser, etc.]
   │
   └── File type detection
```

**Key Architectural Decisions:**

1. **Parser Registry Pattern** – Pluggable parsers registered by file extension
2. **Backward Compatibility** – Existing `video_id` data migrated to `source_id`
3. **Configurable File Types** – UI allows selecting which file types to process
4. **Abstract Base Parser** – Common interface for all document parsers

---

### Terminology Mapping

| Current Term            | New Term                | Notes                                 |
| ----------------------- | ----------------------- | ------------------------------------- |
| `VideoMetadata`         | `SourceMetadata`        | Generic metadata container            |
| `ProcessedVideo`        | `ProcessedDocument`     | Processed content container           |
| `SubtitleEntry`         | `TextEntry`             | Already generic enough, keep          |
| `video_id`              | `source_id`             | Unique identifier for source document |
| `video_ids`             | `source_ids`            | Collection of source identifiers      |
| `video_title`           | `title`                 | Keep as `title`                       |
| `video_metadata`        | `source_metadata`       | Metadata dictionary                   |
| `subtitle_embeddings_*` | `document_embeddings_*` | ChromaDB collection name              |

---

### Data Models

**Before:**

```python
@dataclass
class VideoMetadata:
    video_id: str
    title: str
    date: str
    channel: str

@dataclass
class ProcessedVideo:
    video_id: str
    metadata: VideoMetadata
    chunks: List[Chunk]
```

**After:**

```python
@dataclass
class SourceMetadata:
    source_id: str       # Unique identifier (filename hash, video_id, etc.)
    title: str           # Human-readable title
    date: str            # Date string
    source_type: str     # "srt", "txt", "md", "pdf", etc.
    original_filename: str
    extra: Dict[str, Any] = field(default_factory=dict)  # Format-specific metadata

@dataclass
class ProcessedDocument:
    source_id: str
    metadata: SourceMetadata
    chunks: List[Chunk]
    content_type: str    # MIME type or file extension
```

---

### Parser Interface

```python
from abc import ABC, abstractmethod
from typing import List, Tuple

class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions (e.g., ['.srt', '.sub'])."""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> Tuple[List[TextEntry], SourceMetadata]:
        """Parse document and return text entries with metadata."""
        pass

    @abstractmethod
    def extract_metadata(self, file_path: Path) -> SourceMetadata:
        """Extract metadata from file without full parsing."""
        pass


class ParserRegistry:
    """Registry for document parsers."""

    _parsers: Dict[str, DocumentParser] = {}

    @classmethod
    def register(cls, parser: DocumentParser) -> None:
        for ext in parser.supported_extensions:
            cls._parsers[ext.lower()] = parser

    @classmethod
    def get_parser(cls, file_path: Path) -> Optional[DocumentParser]:
        ext = file_path.suffix.lower()
        return cls._parsers.get(ext)

    @classmethod
    def supported_extensions(cls) -> List[str]:
        return list(cls._parsers.keys())
```

---

### Database Migration

**ChromaDB Metadata Schema Change:**

| Field               | Before   | After                          |
| ------------------- | -------- | ------------------------------ |
| `video_id`          | Required | Renamed to `source_id`         |
| `source_type`       | N/A      | New field (e.g., "srt", "txt") |
| `original_filename` | N/A      | New field                      |

**Migration Strategy:**

1. Create migration script to rename `video_id` → `source_id` in existing collections
2. Add `source_type: "srt"` to all existing entries
3. Backup existing data before migration

---

### Impact Assessment

| Component            | Impact Level | Changes Required                          |
| -------------------- | ------------ | ----------------------------------------- |
| `src/preprocessing/` | **High**     | Core data models, parser interface        |
| `src/vector_store/`  | **Medium**   | Metadata field names                      |
| `src/embeddings/`    | **Low**      | Update ProcessedVideo → ProcessedDocument |
| `src/retrieval/`     | **Medium**   | Filter field names                        |
| `src/clustering/`    | **Low**      | Metadata field names                      |
| `src/ui/`            | **Medium**   | Labels, file upload validation            |
| `scripts/`           | **Medium**   | CLI arguments, variable names             |
| `tests/`             | **Medium**   | Test fixtures, assertions                 |

---

### Risk Assessment

| Risk                   | Impact | Mitigation                    |
| ---------------------- | ------ | ----------------------------- |
| Breaking existing data | High   | Migration script with backup  |
| Test failures          | Medium | Update fixtures incrementally |
| UI regressions         | Medium | Manual testing checklist      |
| Parser compatibility   | Low    | Keep SRT parser as reference  |

---

## Implementation Plan

### 1. [ ] Phase 1: Core Data Model Refactoring

_Description: Update core data structures and create parser abstraction layer._

#### 1.1. [ ] Task: Create abstract parser interface

_Description: Define `DocumentParser` ABC and `ParserRegistry` in new file._

##### 1.1.1. [ ] Sub-Task: Create `src/preprocessing/parser_base.py`

_Description: Implement abstract base class with:_

- `DocumentParser` ABC with `parse()`, `extract_metadata()`, `supported_extensions`
- `ParserRegistry` class with `register()`, `get_parser()`, `supported_extensions()`
- Type definitions for parser outputs

##### 1.1.2. [ ] Sub-Task: Create `src/preprocessing/parsers/__init__.py`

_Description: Create parsers subpackage for organizing format-specific parsers._

#### 1.2. [ ] Task: Refactor metadata classes

_Description: Rename and extend metadata dataclasses._

##### 1.2.1. [ ] Sub-Task: Update `src/preprocessing/metadata.py`

_Description: Rename classes and add new fields:_

- `VideoMetadata` → `SourceMetadata`
- Add `source_type: str` field
- Add `original_filename: str` field
- Add `extra: Dict[str, Any]` for format-specific metadata
- Update `to_dict()` method
- Keep `VideoMetadata` as alias for backward compatibility

##### 1.2.2. [ ] Sub-Task: Update `MetadataExtractor` class

_Description: Generalize to handle multiple file formats:_

- Rename internal references from `video_id` to `source_id`
- Make filename pattern configurable
- Add `source_type` detection from file extension

#### 1.3. [ ] Task: Refactor `ProcessedVideo` to `ProcessedDocument`

_Description: Update main data container class._

##### 1.3.1. [ ] Sub-Task: Update `src/preprocessing/pipeline.py`

_Description:_

- Rename `ProcessedVideo` → `ProcessedDocument`
- Update field `video_id` → `source_id`
- Add `content_type` field
- Keep `ProcessedVideo` as alias for backward compatibility
- Update `PreprocessingPipeline` to use new class names

##### 1.3.2. [ ] Sub-Task: Update `src/preprocessing/__init__.py` exports

_Description: Export new class names while maintaining backward compatibility aliases._

#### 1.4. [ ] Task: Refactor SRT parser to implement interface

_Description: Make existing SRT parser implement the new abstract interface._

##### 1.4.1. [ ] Sub-Task: Move `src/preprocessing/srt_parser.py` to `src/preprocessing/parsers/srt_parser.py`

_Description: Relocate and refactor:_

- Implement `DocumentParser` interface
- Register with `ParserRegistry`
- Keep `SubtitleEntry` as `TextEntry` (already generic)

##### 1.4.2. [ ] Sub-Task: Create `src/preprocessing/parsers/text_parser.py`

_Description: Implement basic plain text parser:_

- Support `.txt` files
- Split by paragraphs or sentences
- Extract metadata from filename

##### 1.4.3. [ ] Sub-Task: Create `src/preprocessing/parsers/markdown_parser.py`

_Description: Implement Markdown parser:_

- Support `.md` files
- Preserve structure (headers, lists)
- Extract title from first H1

---

### 2. [ ] Phase 2: Vector Store Layer Updates

_Description: Update vector store to use new terminology._

#### 2.1. [ ] Task: Update ChromaDB manager

_Description: Rename metadata fields in vector store operations._

##### 2.1.1. [ ] Sub-Task: Update `src/vector_store/chroma_manager.py`

_Description:_

- Rename `video_id` → `source_id` in metadata handling
- Update collection name pattern: `subtitle_embeddings_*` → `document_embeddings_*`
- Add migration helper method

##### 2.1.2. [ ] Sub-Task: Update `src/vector_store/pipeline.py`

_Description:_

- Rename `index_processed_video()` → `index_processed_document()`
- Rename `index_multiple_videos()` → `index_multiple_documents()`
- Update parameter names: `processed_video` → `processed_document`
- Update metadata field names
- Keep old method names as aliases with deprecation warnings

#### 2.2. [ ] Task: Update embedder module

_Description: Update embedding generation to use new types._

##### 2.2.1. [ ] Sub-Task: Update `src/embeddings/embedder.py`

_Description:_

- Update `prepare_for_indexing()` to use `source_metadata`
- Rename `video_metadata` parameter → `source_metadata`

##### 2.2.2. [ ] Sub-Task: Update `src/embeddings/pipeline.py`

_Description:_

- Update imports to use `ProcessedDocument`
- Rename variables from `processed_video` to `processed_document`

---

### 3. [ ] Phase 3: Retrieval & Clustering Updates

_Description: Update search and clustering modules._

#### 3.1. [ ] Task: Update retrieval module

_Description: Rename filter fields and result metadata._

##### 3.1.1. [ ] Sub-Task: Update `src/retrieval/filters.py`

_Description:_

- Rename `video_id` → `source_id` in `SearchFilters` class
- Update filter construction methods

##### 3.1.2. [ ] Sub-Task: Update `src/retrieval/query_engine.py`

_Description:_

- Update metadata field references
- Rename `video_id` → `source_id` in result handling

##### 3.1.3. [ ] Sub-Task: Update `src/retrieval/similarity_search.py`

_Description:_

- Update metadata field names in search results

#### 3.2. [ ] Task: Update clustering module

_Description: Rename metadata fields in clustering._

##### 3.2.1. [ ] Sub-Task: Update `src/clustering/cluster_manager.py`

_Description:_

- Rename `video_ids` → `source_ids` in `ClusterMetadata`
- Update cluster analysis methods

##### 3.2.2. [ ] Sub-Task: Update `src/clustering/cluster_integrator.py`

_Description:_

- Update metadata field references

---

### 4. [ ] Phase 4: UI Layer Updates

_Description: Update Streamlit UI labels and file handling._

#### 4.1. [ ] Task: Update Load Documents page

_Description: Generalize file upload and processing._

##### 4.1.1. [ ] Sub-Task: Update `src/ui/pages/load_documents.py`

_Description:_

- Change file upload to accept multiple extensions (configurable)
- Add file type selector: "SRT", "TXT", "MD", "All supported"
- Update labels: "Video" → "Document", "Subtitle" → "Content"
- Update validation messages
- Use `ParserRegistry.supported_extensions()` for accepted files

##### 4.1.2. [ ] Sub-Task: Add file type configuration

_Description:_

- Add sidebar option to select which file types to process
- Show supported formats dynamically from registry

#### 4.2. [ ] Task: Update PostProcessing page

_Description: Update visualization labels and filters._

##### 4.2.1. [ ] Sub-Task: Update `src/ui/pages/postprocessing.py`

_Description:_

- Rename "Video" → "Source" in all labels
- Update metric labels: "Unique Videos" → "Unique Sources"
- Update chart labels and legends
- Update color-by options: "Video ID" → "Source ID"

#### 4.3. [ ] Task: Update Search page

_Description: Update search filters and result display._

##### 4.3.1. [ ] Sub-Task: Update `src/ui/pages/search.py`

_Description:_

- Rename filter label: "Video ID filter" → "Source filter"
- Update result card: "Video:" → "Source:"
- Make YouTube link conditional (only for SRT/video sources)
- Update metadata display

#### 4.4. [ ] Task: Update UI components

_Description: Update reusable components._

##### 4.4.1. [ ] Sub-Task: Update `src/ui/components/result_card.py`

_Description:_

- Rename `video_id` references → `source_id`
- Make source link conditional based on `source_type`

##### 4.4.2. [ ] Sub-Task: Update `src/ui/state.py`

_Description:_

- Update any video-specific session state keys

##### 4.4.3. [ ] Sub-Task: Update `src/ui/theme.py`

_Description:_

- Update any video-specific labels in constants

---

### 5. [ ] Phase 5: Scripts & CLI Updates

_Description: Update command-line scripts and help text._

#### 5.1. [ ] Task: Update processing script

_Description: Generalize main processing script._

##### 5.1.1. [ ] Sub-Task: Rename and update `scripts/process_subtitles.py`

_Description:_

- Rename file to `scripts/process_documents.py`
- Update CLI argument: `--subtitles-dir` → `--input-dir`
- Add `--file-types` argument
- Update all variable names and messages
- Keep old script as wrapper with deprecation warning

#### 5.2. [ ] Task: Update query script

_Description: Update search script._

##### 5.2.1. [ ] Sub-Task: Update `scripts/query_subtitles.py`

_Description:_

- Rename to `scripts/query_documents.py`
- Update CLI arguments
- Update output labels
- Keep old script as wrapper

#### 5.3. [ ] Task: Update view scripts

_Description: Update visualization scripts._

##### 5.3.1. [ ] Sub-Task: Update `scripts/view_vector_db.py`

_Description:_

- Update labels: "Video ID" → "Source ID"
- Update variable names

##### 5.3.2. [ ] Sub-Task: Update `scripts/explore_vector_db.py`

_Description:_

- Update metadata field references
- Update display labels

---

### 6. [ ] Phase 6: Database Migration

_Description: Create and execute data migration._

#### 6.1. [ ] Task: Create migration script

_Description: Script to migrate existing ChromaDB data._

##### 6.1.1. [ ] Sub-Task: Create `scripts/migrate_to_documents.py`

_Description: Migration script that:_

1. Backs up existing `data/vector_db/` directory
2. Reads all documents from collection
3. Updates metadata: `video_id` → `source_id`, adds `source_type: "srt"`
4. Optionally renames collection from `subtitle_embeddings_*` to `document_embeddings_*`
5. Validates migration
6. Provides rollback option

##### 6.1.2. [ ] Sub-Task: Create migration documentation

_Description: Add `docs/migration_guide.md` with:_

- Step-by-step migration instructions
- Rollback procedures
- Troubleshooting guide

---

### 7. [ ] Phase 7: Test Updates

_Description: Update test fixtures and assertions._

#### 7.1. [ ] Task: Update test fixtures

_Description: Update test data to use new terminology._

##### 7.1.1. [ ] Sub-Task: Update `tests/fixtures/`

_Description:_

- Update sample metadata with `source_id`
- Add `source_type` fields to fixtures

##### 7.1.2. [ ] Sub-Task: Update `tests/test_preprocessing.py`

_Description:_

- Update class name references
- Update field name assertions
- Add tests for new parser interface

##### 7.1.3. [ ] Sub-Task: Update `tests/test_retrieval.py`

_Description:_

- Update filter field names
- Update result assertions

##### 7.1.4. [ ] Sub-Task: Update `tests/test_clustering.py`

_Description:_

- Update metadata field references

##### 7.1.5. [ ] Sub-Task: Update `tests/test_integration.py`

_Description:_

- Update end-to-end test scenarios
- Add tests for new file types

#### 7.2. [ ] Task: Add new tests

_Description: Tests for new functionality._

##### 7.2.1. [ ] Sub-Task: Create `tests/test_parser_registry.py`

_Description: Tests for:_

- Parser registration
- File type detection
- Parser retrieval by extension

##### 7.2.2. [ ] Sub-Task: Create `tests/test_text_parser.py`

_Description: Tests for plain text parser._

##### 7.2.3. [ ] Sub-Task: Create `tests/test_markdown_parser.py`

_Description: Tests for Markdown parser._

---

### 8. [ ] Phase 8: Documentation Updates

_Description: Update all documentation._

#### 8.1. [ ] Task: Update README files

_Description: Update project documentation._

##### 8.1.1. [ ] Sub-Task: Update `README.md`

_Description:_

- Update project description
- Update feature list
- Update usage examples

##### 8.1.2. [ ] Sub-Task: Update `README_STREAMLIT.md`

_Description:_

- Update UI documentation
- Update screenshots descriptions

#### 8.2. [ ] Task: Update code documentation

_Description: Update docstrings and comments._

##### 8.2.1. [ ] Sub-Task: Update all module docstrings

_Description: Replace video/subtitle references with generic document terminology._

##### 8.2.2. [ ] Sub-Task: Update inline comments

_Description: Update comments throughout codebase._

---

### 9. [ ] Phase 9: Cleanup & Finalization

_Description: Remove deprecated code and finalize._

#### 9.1. [ ] Task: Remove backward compatibility aliases

_Description: After migration period, remove deprecated names._

##### 9.1.1. [ ] Sub-Task: Remove `ProcessedVideo` alias

_Description: Remove alias after confirming all code updated._

##### 9.1.2. [ ] Sub-Task: Remove `VideoMetadata` alias

_Description: Remove alias after confirming all code updated._

##### 9.1.3. [ ] Sub-Task: Remove old script wrappers

_Description: Remove deprecated script names._

#### 9.2. [ ] Task: Final validation

_Description: Comprehensive testing and validation._

##### 9.2.1. [ ] Sub-Task: Run full test suite

_Description: Ensure all tests pass._

##### 9.2.2. [ ] Sub-Task: Manual UI testing

_Description: Test all UI functionality._

##### 9.2.3. [ ] Sub-Task: Performance validation

_Description: Ensure no performance regression._

---

## Appendix: File Changes Summary

### Files to Create

| File                                           | Description               |
| ---------------------------------------------- | ------------------------- |
| `src/preprocessing/parser_base.py`             | Abstract parser interface |
| `src/preprocessing/parsers/__init__.py`        | Parsers subpackage        |
| `src/preprocessing/parsers/srt_parser.py`      | Moved SRT parser          |
| `src/preprocessing/parsers/text_parser.py`     | Plain text parser         |
| `src/preprocessing/parsers/markdown_parser.py` | Markdown parser           |
| `scripts/process_documents.py`                 | Renamed processing script |
| `scripts/query_documents.py`                   | Renamed query script      |
| `scripts/migrate_to_documents.py`              | Migration script          |
| `docs/migration_guide.md`                      | Migration documentation   |
| `tests/test_parser_registry.py`                | Parser registry tests     |
| `tests/test_text_parser.py`                    | Text parser tests         |
| `tests/test_markdown_parser.py`                | Markdown parser tests     |

### Files to Modify (Major Changes)

| File                                 | Changes                               |
| ------------------------------------ | ------------------------------------- |
| `src/preprocessing/metadata.py`      | Rename classes, add fields            |
| `src/preprocessing/pipeline.py`      | Rename ProcessedVideo, update methods |
| `src/preprocessing/__init__.py`      | Update exports                        |
| `src/vector_store/chroma_manager.py` | Update metadata fields                |
| `src/vector_store/pipeline.py`       | Rename methods                        |
| `src/retrieval/filters.py`           | Rename filter fields                  |
| `src/retrieval/query_engine.py`      | Update metadata references            |
| `src/ui/pages/load_documents.py`     | Update labels, file handling          |
| `src/ui/pages/postprocessing.py`     | Update labels                         |
| `src/ui/pages/search.py`             | Update labels, filters                |

### Files to Modify (Minor Changes)

| File                                | Changes                     |
| ----------------------------------- | --------------------------- |
| `src/embeddings/embedder.py`        | Rename parameters           |
| `src/embeddings/pipeline.py`        | Update imports              |
| `src/clustering/cluster_manager.py` | Rename fields               |
| `src/ui/components/result_card.py`  | Update labels               |
| `scripts/view_vector_db.py`         | Update labels               |
| `scripts/explore_vector_db.py`      | Update labels               |
| All test files                      | Update fixtures, assertions |

---

## Estimated Effort

| Phase                           | Estimated Hours | Priority |
| ------------------------------- | --------------- | -------- |
| Phase 1: Core Data Models       | 6               | Critical |
| Phase 2: Vector Store           | 3               | Critical |
| Phase 3: Retrieval & Clustering | 2               | High     |
| Phase 4: UI Updates             | 4               | High     |
| Phase 5: Scripts & CLI          | 2               | Medium   |
| Phase 6: Database Migration     | 3               | Critical |
| Phase 7: Test Updates           | 4               | High     |
| Phase 8: Documentation          | 2               | Medium   |
| Phase 9: Cleanup                | 2               | Low      |
| **Total**                       | **28 hours**    |          |

---

## Success Criteria

- [ ] All tests pass with new terminology
- [ ] Existing SRT data works after migration
- [ ] New file types (TXT, MD) can be processed end-to-end
- [ ] UI shows generic labels ("Source" instead of "Video")
- [ ] CLI scripts work with new argument names
- [ ] No performance regression
- [ ] Documentation reflects new capabilities
