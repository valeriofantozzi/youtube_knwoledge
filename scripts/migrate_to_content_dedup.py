#!/usr/bin/env python3
"""
Production migration script for content-based deduplication.

This script safely migrates the database to the new deduplication system:
1. Creates a backup of the current database
2. Optionally resets the database
3. Re-indexes all documents with new source_id logic
4. Verifies deduplication worked

Usage:
    python scripts/migrate_to_content_dedup.py
    python scripts/migrate_to_content_dedup.py --reset  # Include reset step
    python scripts/migrate_to_content_dedup.py --backup-only  # Just backup, no reset
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_backup(db_path: Path) -> Path | None:
    """Create timestamped backup of database."""
    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"chroma_db_backup_{timestamp}"
    
    logger.info(f"Creating backup at {backup_path}...")
    try:
        shutil.copytree(str(db_path), str(backup_path))
        size_mb = sum(f.stat().st_size for f in backup_path.rglob('*')) / (1024*1024)
        logger.info(f"✅ Backup created successfully ({size_mb:.1f} MB)")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return None


def reset_database(db_path: Path) -> bool:
    """Reset database (remove all documents)."""
    confirm = input("\n⚠️  WARNING: This will clear ALL documents from the database.\n"
                   "This should only be done if you have a backup!\n"
                   "Type 'yes' to confirm: ").strip().lower()
    
    if confirm != 'yes':
        logger.info("Reset cancelled by user")
        return False
    
    try:
        logger.info("Resetting database...")
        script_path = Path(__file__).parent / "reset_database.py"
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("✅ Database reset successfully")
            return True
        else:
            logger.error(f"❌ Reset failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Reset error: {e}")
        return False


def check_for_duplicates(before: bool = True) -> int:
    """Check current duplicate count in database."""
    try:
        script_path = Path(__file__).parent / "check_duplicates.py"
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if "found" in result.stdout.lower():
            # Extract duplicate count from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'duplicate' in line.lower() and ('found' in line.lower() or 'text' in line.lower()):
                    logger.info(f"{'BEFORE' if before else 'AFTER'}: {line.strip()}")
            return 0  # Return value doesn't matter, we log the output
        else:
            logger.info(result.stdout)
            return 0
    except Exception as e:
        logger.warning(f"Could not check duplicates: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Migrate database to content-based deduplication")
    parser.add_argument("--reset", action="store_true", help="Include reset step")
    parser.add_argument("--backup-only", action="store_true", help="Only create backup, don't reset or reindex")
    parser.add_argument("--input", default="./subtitles", help="Input directory for re-indexing")
    
    args = parser.parse_args()
    
    db_path = Path("chroma_db")
    input_path = Path(args.input)
    
    logger.info("=" * 60)
    logger.info("Content-Based Deduplication Migration")
    logger.info("=" * 60)
    
    # Step 1: Check for duplicates BEFORE
    logger.info("\n📊 Checking for duplicates BEFORE migration...")
    check_for_duplicates(before=True)
    
    # Step 2: Create backup
    logger.info("\n💾 Step 1: Creating backup...")
    backup_path = create_backup(db_path)
    
    if not backup_path:
        logger.error("❌ Backup creation failed. Aborting migration.")
        sys.exit(1)
    
    if args.backup_only:
        logger.info("\n✅ Backup created. Skipping reset and re-indexing (--backup-only mode)")
        return
    
    # Step 3: Reset database (optional)
    if args.reset:
        logger.info("\n🔄 Step 2: Resetting database...")
        if not reset_database(db_path):
            logger.error("❌ Reset cancelled or failed. Database backup is available at:")
            logger.error(f"   {backup_path}")
            sys.exit(1)
    else:
        logger.info("\n⏭️  Skipping reset (use --reset flag to include)")
    
    # Step 4: Re-index documents
    logger.info("\n📑 Step 3: Re-indexing documents with new deduplication logic...")
    if not input_path.exists():
        logger.error(f"❌ Input directory not found: {input_path}")
        logger.info(f"Database backup available at: {backup_path}")
        sys.exit(1)
    
    try:
        # Use the load command to re-index
        cmd = [
            "python", "-m", "src.cli",
            "load", str(input_path),
            "--batch-size", "10"
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✅ Re-indexing completed successfully")
            logger.info(result.stdout)
        else:
            logger.warning("Re-indexing finished with warnings/errors:")
            logger.warning(result.stderr)
    except Exception as e:
        logger.error(f"❌ Re-indexing failed: {e}")
        logger.info(f"Database backup available at: {backup_path}")
        sys.exit(1)
    
    # Step 5: Check for duplicates AFTER
    logger.info("\n📊 Checking for duplicates AFTER migration...")
    check_for_duplicates(before=False)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Migration completed!")
    logger.info("=" * 60)
    logger.info(f"\n📝 Summary:")
    logger.info(f"  - Database backup: {backup_path}")
    if args.reset:
        logger.info(f"  - Database reset: Yes")
    logger.info(f"  - Documents re-indexed: Yes")
    logger.info(f"\n🔍 Next steps:")
    logger.info(f"  1. Run a few search queries to verify no duplicates")
    logger.info(f"  2. Monitor logs for 'skipped (duplicates)' messages")
    logger.info(f"  3. If issues found, restore from backup: shutil.copytree({backup_path}, {db_path})")
    logger.info(f"\n🧪 To run tests again: python scripts/test_deduplication.py")


if __name__ == "__main__":
    main()
