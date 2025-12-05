#!/usr/bin/env python3
"""Backup the current vector database."""

import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config


def backup_database():
    """Create a timestamped backup of the database."""
    config = get_config()
    db_path = Path(config.VECTOR_DB_PATH)
    
    if not db_path.exists():
        print("❌ Database not found, nothing to backup")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"chroma_db_backup_{timestamp}"
    
    print(f"Creating backup: {backup_path}")
    print(f"Source: {db_path}")
    print(f"Size: {sum(f.stat().st_size for f in db_path.rglob('*')) / (1024*1024):.1f} MB")
    
    try:
        shutil.copytree(db_path, backup_path)
        print(f"✅ Backup created successfully")
        print(f"📂 Location: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False


if __name__ == "__main__":
    success = backup_database()
    sys.exit(0 if success else 1)
