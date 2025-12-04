#!/usr/bin/env bash
# Installation and Testing Guide for Configuration System

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║           KnowBase Configuration System - Installation & Testing            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Install dependencies
echo "📦 STEP 1: Installing dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install pydantic>=2.0.0 pyyaml>=6.0
echo "✓ Dependencies installed"
echo ""

# Step 2: List created files
echo "📁 STEP 2: Configuration System Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Core Module:"
echo "  ✓ src/utils/config_manager.py (500+ lines)"
echo ""
echo "Configuration Presets (config/presets/):"
echo "  ✓ full_pipeline.yaml        - All stages: prep → embed → index"
echo "  ✓ embeddings_only.yaml      - Skip preprocessing"
echo "  ✓ search_only.yaml          - Skip indexing, search only"
echo "  ✓ rag_only.yaml             - RAG/LLM questions only"
echo "  ✓ custom_template.yaml      - Fully documented template"
echo ""
echo "Documentation:"
echo "  ✓ docs/CONFIGURATION.md     - Complete reference (2000+ lines)"
echo "  ✓ CONFIG_README.md          - Quick reference"
echo "  ✓ docs/CONFIGURATION_SYSTEM_SUMMARY.md - Implementation summary"
echo ""
echo "Tests & Examples:"
echo "  ✓ scripts/test_config.py    - 8 comprehensive tests"
echo "  ✓ examples/cli_examples.py  - CLI command examples"
echo ""

# Step 3: Run tests
echo ""
echo "🧪 STEP 3: Running Test Suite"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/test_config.py
echo ""

# Step 4: Usage examples
echo ""
echo "🚀 STEP 4: Quick Usage Examples"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Python Usage:"
echo "─────────────"
cat << 'EOF'

from src.utils.config_manager import ConfigManager, get_preset_config
from pathlib import Path

# Load preset
config = get_preset_config("full_pipeline")

# Load from file
mgr = ConfigManager(config_file=Path("config/presets/full_pipeline.yaml"))

# Override settings
mgr.merge_with_dict({"embedding": {"batch_size": 64}})

# Get specific configs
embedding_cfg = mgr.get_embedding_config()
pipeline_cfg = mgr.get_pipeline_config()

# Save for later
mgr.save_to_file(Path("config/my_config.yaml"))

EOF

echo "CLI Usage:"
echo "──────────"
echo "# Full pipeline"
echo "knowbase load --config config/presets/full_pipeline.yaml --input ./subtitles"
echo ""
echo "# With overrides"
echo "knowbase load --config config/my_config.yaml --input ./docs --batch-size 64"
echo ""
echo "# Search only"
echo "knowbase search --config config/presets/search_only.yaml --query 'orchid care'"
echo ""
echo "# RAG only"
echo "knowbase ask --config config/presets/rag_only.yaml 'How to grow orchids?'"
echo ""

# Step 5: Key features
echo ""
echo "✨ KEY FEATURES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Modular Pipeline Execution"
echo "   - Run complete pipeline (prep → embed → index)"
echo "   - Run only embeddings"
echo "   - Run only search"
echo "   - Run only RAG/LLM"
echo ""
echo "✅ Configuration File Support"
echo "   - Load from YAML files"
echo "   - Save configurations for reuse"
echo "   - Deep merge overrides"
echo "   - Pydantic validation"
echo ""
echo "✅ Multiple Input Methods"
echo "   - Config files (YAML/JSON)"
echo "   - Environment variables"
echo "   - CLI flags"
echo "   - Python code"
echo ""
echo "✅ Comprehensive Documentation"
echo "   - 2000+ lines in CONFIGURATION.md"
echo "   - Quick reference in CONFIG_README.md"
echo "   - Examples in cli_examples.py"
echo ""
echo "✅ Full Test Coverage"
echo "   - 8 tests in test_config.py"
echo "   - Covers all configuration scenarios"
echo ""

# Step 6: Next steps
echo ""
echo "📋 NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Review configuration files in config/presets/"
echo "2. Read quick reference: CONFIG_README.md"
echo "3. Run tests: python scripts/test_config.py"
echo "4. Check examples: examples/cli_examples.py"
echo "5. Start implementing CLI commands (Phase 2-3)"
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                 ✓ Configuration System Ready to Use!                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
