#!/usr/bin/env python3
"""Quick test of content-based deduplication system."""

import sys
import hashlib
import tempfile
from pathlib import Path
from typing import Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.metadata_extractor import MetadataExtractor

def test_content_hash():
    """Test that content hash generates correct source_ids and content_hashes."""
    print("🔬 Testing content-based deduplication...\n")
    
    extractor = MetadataExtractor()
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1: Two files with identical content
        print("Test 1: Identical content files")
        file1 = tmpdir / "20231111_Channel1_VideoTitle1.en.srt"
        file2 = tmpdir / "20231112_Channel2_VideoTitle2.en.srt"
        
        test_content = "This is identical content.\nSame in both files."
        
        with open(file1, 'w') as f:
            f.write(test_content)
        with open(file2, 'w') as f:
            f.write(test_content)
        
        source_id_1, content_hash_1 = extractor._generate_source_id_from_content(file1)
        source_id_2, content_hash_2 = extractor._generate_source_id_from_content(file2)
        
        print(f"  File 1: {file1.name}")
        print(f"  Source ID 1: {source_id_1}")
        print(f"  Content Hash 1: {content_hash_1}")
        print(f"  File 2: {file2.name}")
        print(f"  Source ID 2: {source_id_2}")
        print(f"  Content Hash 2: {content_hash_2}")
        
        # Verify Source IDs are still different (contain filename)
        if source_id_1 != source_id_2:
             print(f"  ✅ PASS: Source IDs are different (as expected for different filenames)")
        else:
             print(f"  ❌ FAIL: Source IDs are identical (unexpected)")
             return False

        # Verify Content Hashes are IDENTICAL
        if content_hash_1 == content_hash_2:
            print(f"  ✅ PASS: Identical content → same content hash ({content_hash_1})")
        else:
            print(f"  ❌ FAIL: Identical content → different content hashes")
            print(f"  Hash 1: {content_hash_1}")
            print(f"  Hash 2: {content_hash_2}\n")
            return False
            
        print("")
        
        # Test 2: Two files with different content
        print("Test 2: Different content files")
        file3 = tmpdir / "20231113_Channel3_VideoTitle3.en.srt"
        different_content = "This is different content.\nNot the same."
        
        with open(file3, 'w') as f:
            f.write(different_content)
        
        source_id_3, content_hash_3 = extractor._generate_source_id_from_content(file3)
        
        print(f"  File 3: {file3.name}")
        print(f"  Source ID 3: {source_id_3}")
        print(f"  Content Hash 3: {content_hash_3}")
        
        if content_hash_1 != content_hash_3:
            print(f"  ✅ PASS: Different content → different content hashes")
            print(f"  Hash 1: {content_hash_1}")
            print(f"  Hash 3: {content_hash_3}\n")
        else:
            print(f"  ❌ FAIL: Different content → same content hashes\n")
            return False
        
        # Test 3: Verify hash format
        print("Test 3: Hash format verification")
        
        if len(content_hash_1) == 64 and all(c in '0123456789abcdef' for c in content_hash_1):
             print(f"  ✅ PASS: Content Hash format is correct (64 hex chars)")
        else:
             print(f"  ❌ FAIL: Content Hash format is incorrect: {content_hash_1}")
             return False

        # Test 4: Deterministic hash
        print("Test 4: Deterministic hash (same content → same hash)")
        source_id_1_again, content_hash_1_again = extractor._generate_source_id_from_content(file1)
        
        if content_hash_1 == content_hash_1_again:
            print(f"  ✅ PASS: Hash is deterministic\n")
        else:
            print(f"  ❌ FAIL: Hash is not deterministic\n")
            return False
            
        # Test 5: Verify SourceMetadata extraction
        print("Test 5: SourceMetadata extraction")
        metadata = extractor.extract_from_filename(file1)
        if metadata.content_hash == content_hash_1:
             print(f"  ✅ PASS: content_hash correctly propagated to SourceMetadata")
        else:
             print(f"  ❌ FAIL: SourceMetadata.content_hash mismatch")
             print(f"  Expected: {content_hash_1}")
             print(f"  Got: {metadata.content_hash}")
             return False

    
    print("=" * 60)
    print("✅ All tests passed! Deduplication system logic is correct.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_content_hash()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
