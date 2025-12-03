#!/usr/bin/env python
"""
Clear Checkpoints Script

Utility script to clear embedding generation checkpoints.
Useful when switching models or when checkpoints are corrupted.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.utils.logger import get_default_logger


def clear_all_checkpoints(checkpoint_dir: Path) -> int:
    """
    Clear all checkpoint files.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Number of checkpoints cleared
    """
    if not checkpoint_dir.exists():
        return 0
    
    checkpoint_files = list(checkpoint_dir.glob("*_checkpoint.pkl"))
    count = 0
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint_file.unlink()
            count += 1
        except Exception as e:
            print(f"Warning: Failed to delete {checkpoint_file.name}: {e}")
    
    return count


def clear_specific_checkpoint(checkpoint_dir: Path, source_id: str) -> bool:
    """
    Clear checkpoint for a specific document.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        source_id: Document source ID
        
    Returns:
        True if checkpoint was found and cleared
    """
    checkpoint_file = checkpoint_dir / f"{source_id}_checkpoint.pkl"
    
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            return True
        except Exception as e:
            print(f"Error: Failed to delete checkpoint for {source_id}: {e}")
            return False
    
    return False


def list_checkpoints(checkpoint_dir: Path) -> list:
    """
    List all checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        List of checkpoint file paths
    """
    if not checkpoint_dir.exists():
        return []
    
    return sorted(checkpoint_dir.glob("*_checkpoint.pkl"))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Clear embedding generation checkpoints"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Checkpoint directory (default: from config)"
    )
    
    parser.add_argument(
        "--source-id",
        type=str,
        help="Clear checkpoint for specific document source ID only"
    )
    
    # Backward compatibility alias
    parser.add_argument(
        "--video-id",
        type=str,
        dest="source_id",
        help="(Deprecated: use --source-id) Clear checkpoint for specific document"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all checkpoints without deleting"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    logger = get_default_logger()
    
    # Determine checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(getattr(config, "CHECKPOINT_DIR", "./data/checkpoints"))
    
    # List mode
    if args.list:
        checkpoints = list_checkpoints(checkpoint_dir)
        
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            return 0
        
        print(f"Found {len(checkpoints)} checkpoint(s) in {checkpoint_dir}:\n")
        
        for checkpoint in checkpoints:
            size_kb = checkpoint.stat().st_size / 1024
            source_id = checkpoint.stem.replace("_checkpoint", "")
            print(f"  - {source_id}: {size_kb:.1f} KB")
        
        return 0
    
    # Clear specific checkpoint
    if args.source_id:
        print(f"Clearing checkpoint for document: {args.source_id}")
        
        if not args.force:
            response = input("Are you sure? This cannot be undone. (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0
        
        if clear_specific_checkpoint(checkpoint_dir, args.source_id):
            print(f"✓ Cleared checkpoint for {args.source_id}")
            logger.info(f"Cleared checkpoint for document {args.source_id}")
            return 0
        else:
            print(f"✗ No checkpoint found for {args.source_id}")
            return 1
    
    # Clear all checkpoints
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return 0
    
    print(f"Found {len(checkpoints)} checkpoint(s) in {checkpoint_dir}")
    
    if not args.force:
        response = input("Clear all checkpoints? This cannot be undone. (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    count = clear_all_checkpoints(checkpoint_dir)
    print(f"✓ Cleared {count} checkpoint(s)")
    logger.info(f"Cleared {count} checkpoints from {checkpoint_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
