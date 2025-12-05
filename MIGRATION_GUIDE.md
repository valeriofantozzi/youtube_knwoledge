# Quick Migration Guide - Content-Based Deduplication

## 🎯 Goal

Migrate your knowbase database from timestamp-based to content-based deduplication, eliminating duplicate chunks in search results.

## ⚠️ Prerequisites

- All current work is committed to git
- You have write access to the `chroma_db/` directory
- You have 2-3x the current database size in disk space for backup

## 🚀 Quick Start (3 Commands)

### Option A: Safe Migration (Recommended)

```bash
# Step 1: Create backup only (no database changes yet)
python scripts/migrate_to_content_dedup.py --backup-only

# Step 2: Check results BEFORE migration
python scripts/check_duplicates.py

# Step 3: Run full migration with reset and re-indexing
python scripts/migrate_to_content_dedup.py --reset --input ./subtitles
```

### Option B: Minimal Downtime

```bash
# This keeps existing data while adding new indexed documents
# (not recommended if database already has duplicates)
python scripts/migrate_to_content_dedup.py --input ./subtitles
```

## 📊 Monitoring Progress

During migration, watch for:

- **"Creating backup..."** → Should complete in seconds/minutes
- **"Resetting database..."** → Clears old documents
- **"Re-indexing documents..."** → Loads documents with new dedup logic
- **"skipped (duplicates)"** → Shows how many duplicates were prevented

## ✅ Verification

After migration completes:

```bash
# Check for remaining duplicates (should be 0 or very low)
python scripts/check_duplicates.py

# Run unit tests
python scripts/test_deduplication.py

# Try a search query and verify no duplicate results
# (Open streamlit app and search)
```

## 🔄 Rollback (If Needed)

If something goes wrong:

```bash
# Find your backup
ls -lah chroma_db_backup_*

# Restore from backup
python -c "
import shutil
shutil.rmtree('chroma_db')
shutil.copytree('chroma_db_backup_YYYYMMDD_HHMMSS', 'chroma_db')
print('✅ Database restored from backup')
"
```

## 📈 Expected Results

### Before Migration

- Database contains many duplicate chunks
- Search results show identical text from multiple "sources"
- Example: Standard intro text appears 3+ times in different videos

### After Migration

- Duplicate chunks are prevented at indexing time
- Search results show unique content
- Database size may decrease 40-60% (less duplicates stored)
- Re-indexing documents shows "X skipped (duplicates)"

## 🔧 Troubleshooting

### Q: Migration takes too long?

A: It's re-indexing all documents. This is normal and may take 30-60 minutes for large databases.

### Q: "Permission denied" on backup?

A: Ensure you have write access to the directory containing `chroma_db/`

### Q: Can I interrupt the migration?

A: It's safe to interrupt. The backup is already created, and you can restore from it anytime.

### Q: Do I need to change my application code?

A: No! The deduplication works transparently. Your app just benefits from fewer duplicates.

## 📝 What Changed Under the Hood

1. **Source ID Generation**: Now uses SHA-256 hash of file content (was: timestamp)
2. **Deduplication**: Prevents duplicate `(source_id, chunk_index)` pairs at indexing
3. **Generic Logic**: Removed all video_id-specific code (works for any document type)

## 🎓 Learn More

See `DEDUPLICATION_IMPLEMENTATION.md` for technical details, architecture, and design decisions.

---

**Status**: All implementation complete and tested ✅
**Ready to migrate**: Yes
**Estimated duration**: 30-120 minutes depending on database size
