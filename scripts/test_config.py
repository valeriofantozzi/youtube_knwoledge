#!/usr/bin/env python3
"""
Test script for configuration system.

Run with: python scripts/test_config.py
"""

import sys
from pathlib import Path
import json
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_manager import (
    ConfigManager,
    get_preset_config,
    CompleteConfig,
    EmbeddingConfig,
    PreprocessingConfig,
)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_preset_configs() -> None:
    """Test loading all preset configurations."""
    print_section("TEST 1: Preset Configurations")

    presets = ["full_pipeline", "embeddings_only", "search_only", "rag_only"]

    for preset in presets:
        try:
            config = get_preset_config(preset)  # type: ignore
            print(f"✓ {preset:20} - Config name: {config.config_name}")
            print(f"  Description: {config.description}")
            print(
                f"  Pipeline: prep={config.pipeline.run_preprocessing}, "
                f"emb={config.pipeline.run_embedding}, "
                f"ret={config.pipeline.run_retrieval}, "
                f"rag={config.pipeline.run_ai_search}"
            )
        except Exception as e:
            print(f"✗ {preset:20} - FAILED: {e}")


def test_config_manager_default() -> None:
    """Test ConfigManager with defaults."""
    print_section("TEST 2: ConfigManager with Defaults")

    try:
        mgr = ConfigManager()
        print(f"✓ Created ConfigManager")
        print(f"  Model: {mgr.config.embedding.model_name}")
        print(f"  Device: {mgr.config.embedding.device}")
        print(f"  Batch size: {mgr.config.embedding.batch_size}")
        print(f"  Chunk size: {mgr.config.preprocessing.chunk_size}")
        print(f"  DB path: {mgr.config.vector_store.db_path}")
    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_load_from_file() -> None:
    """Test loading config from YAML file."""
    print_section("TEST 3: Load from YAML File")

    config_file = Path(__file__).parent.parent / "config/presets/full_pipeline.yaml"

    if not config_file.exists():
        print(f"✗ Config file not found: {config_file}")
        return

    try:
        mgr = ConfigManager(config_file=config_file)
        print(f"✓ Loaded from {config_file.name}")
        print(f"  Config name: {mgr.config.config_name}")
        print(f"  Model: {mgr.config.embedding.model_name}")
        print(f"  Run pipelines:")
        print(f"    - Preprocessing: {mgr.config.pipeline.run_preprocessing}")
        print(f"    - Embedding: {mgr.config.pipeline.run_embedding}")
        print(f"    - Indexing: {mgr.config.pipeline.run_indexing}")
        print(f"    - Retrieval: {mgr.config.pipeline.run_retrieval}")
        print(f"    - RAG: {mgr.config.pipeline.run_ai_search}")
    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_merge_overrides() -> None:
    """Test merging configuration with overrides."""
    print_section("TEST 4: Merge Overrides")

    try:
        mgr = ConfigManager()
        print(f"✓ Created ConfigManager")

        # Show original
        print(f"\n  Before merge:")
        print(f"    Model: {mgr.config.embedding.model_name}")
        print(f"    Batch size: {mgr.config.embedding.batch_size}")
        print(f"    Chunk size: {mgr.config.preprocessing.chunk_size}")

        # Merge overrides
        overrides = {
            "embedding": {"model_name": "google/embeddinggemma-300m", "batch_size": 64},
            "preprocessing": {"chunk_size": 256},
        }
        mgr.merge_with_dict(overrides)
        print(f"\n  After merge with {overrides}:")
        print(f"    Model: {mgr.config.embedding.model_name}")
        print(f"    Batch size: {mgr.config.embedding.batch_size}")
        print(f"    Chunk size: {mgr.config.preprocessing.chunk_size}")

    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_save_and_load() -> None:
    """Test saving and loading configuration."""
    print_section("TEST 5: Save and Load Config")

    try:
        # Create and customize config
        mgr = ConfigManager()
        mgr.merge_with_dict(
            {
                "config_name": "test_config",
                "embedding": {"batch_size": 96},
                "preprocessing": {"chunk_size": 384},
            }
        )

        # Save to file
        test_file = Path("/tmp/test_knowbase_config.yaml")
        mgr.save_to_file(test_file)
        print(f"✓ Saved config to {test_file}")

        # Load back
        mgr2 = ConfigManager(config_file=test_file)
        print(f"✓ Loaded config from {test_file}")
        print(f"  Config name: {mgr2.config.config_name}")
        print(f"  Batch size: {mgr2.config.embedding.batch_size}")
        print(f"  Chunk size: {mgr2.config.preprocessing.chunk_size}")

        # Verify they match
        assert mgr2.config.embedding.batch_size == 96
        assert mgr2.config.preprocessing.chunk_size == 384
        print(f"✓ Verified saved config matches")

        # Cleanup
        test_file.unlink()

    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_config_to_formats() -> None:
    """Test converting config to different formats."""
    print_section("TEST 6: Config Format Conversions")

    try:
        mgr = ConfigManager()

        # To dict
        config_dict = mgr.to_dict(exclude_none=True)
        print(f"✓ to_dict() - Keys: {list(config_dict.keys())}")

        # To JSON
        config_json = mgr.to_json()
        json_obj = json.loads(config_json)
        print(f"✓ to_json() - {len(config_json)} chars")

        # To YAML
        config_yaml = mgr.to_yaml()
        print(f"✓ to_yaml() - {len(config_yaml)} chars")

    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_get_specific_configs() -> None:
    """Test getting specific configuration sections."""
    print_section("TEST 7: Get Specific Configs")

    try:
        mgr = ConfigManager()

        emb_config = mgr.get_embedding_config()
        print(f"✓ get_embedding_config() - Model: {emb_config.model_name}")

        prep_config = mgr.get_preprocessing_config()
        print(f"✓ get_preprocessing_config() - Chunk size: {prep_config.chunk_size}")

        vs_config = mgr.get_vector_store_config()
        print(f"✓ get_vector_store_config() - DB path: {vs_config.db_path}")

        ret_config = mgr.get_retrieval_config()
        print(f"✓ get_retrieval_config() - Top K: {ret_config.top_k}")

        ais_config = mgr.get_ai_search_config()
        print(f"✓ get_ai_search_config() - LLM: {ais_config.llm_model}")

        clust_config = mgr.get_clustering_config()
        print(f"✓ get_clustering_config() - Min cluster size: {clust_config.min_cluster_size}")

        pipe_config = mgr.get_pipeline_config()
        print(
            f"✓ get_pipeline_config() - Verbose: {pipe_config.verbose}, "
            f"Log level: {pipe_config.log_level}"
        )

    except Exception as e:
        print(f"✗ FAILED: {e}")


def test_validation() -> None:
    """Test Pydantic validation."""
    print_section("TEST 8: Pydantic Validation")

    try:
        # Valid config
        config = EmbeddingConfig(
            model_name="test-model",
            batch_size=32,
            device="cuda",
            precision="fp16",
        )
        print(f"✓ Valid EmbeddingConfig created")

        # Invalid batch size
        try:
            bad_config = EmbeddingConfig(batch_size=0)  # type: ignore
            print(f"✗ Should have failed on batch_size=0")
        except ValueError:
            print(f"✓ Validation caught invalid batch_size=0")

        # Invalid device
        try:
            bad_config = EmbeddingConfig(device="gpu")  # type: ignore
            print(f"✗ Should have failed on device='gpu'")
        except ValueError:
            print(f"✓ Validation caught invalid device='gpu'")

    except Exception as e:
        print(f"✗ FAILED: {e}")


def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  KnowBase Configuration System Tests")
    print("=" * 70)

    test_preset_configs()
    test_config_manager_default()
    test_load_from_file()
    test_merge_overrides()
    test_save_and_load()
    test_config_to_formats()
    test_get_specific_configs()
    test_validation()

    print("\n" + "=" * 70)
    print("  ✓ All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
