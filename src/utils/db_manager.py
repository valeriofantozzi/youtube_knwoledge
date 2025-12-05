import os
import shutil
from pathlib import Path
from typing import List, Optional

class DatabaseManager:
    def __init__(self, base_path: str = "./data/vector_dbs", legacy_path: str = "./data/vector_db"):
        self.base_path = Path(base_path)
        self.legacy_path = Path(legacy_path)
        self.active_db_file = self.base_path / ".active_db"
        
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Migration: if legacy path exists and default doesn't, move it
        default_db_path = self.base_path / "default"
        if self.legacy_path.exists() and not default_db_path.exists():
            print(f"Migrating legacy database from {self.legacy_path} to {default_db_path}...")
            shutil.move(str(self.legacy_path), str(default_db_path))
            # Create a symlink or just leave it? 
            # If we move it, the old path is gone. 
            # But config might still point to it if we don't update config.
        
        # Ensure default db exists if migration didn't happen
        if not default_db_path.exists():
             default_db_path.mkdir(parents=True, exist_ok=True)

    def list_databases(self) -> List[str]:
        """List all available databases."""
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    def create_database(self, name: str) -> bool:
        """Create a new database."""
        # Validate name (simple alphanumeric check)
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Database name must contain only alphanumeric characters, underscores, or hyphens.")
            
        db_path = self.base_path / name
        if db_path.exists():
            return False
        db_path.mkdir(parents=True, exist_ok=True)
        return True

    def remove_database(self, name: str) -> bool:
        """Remove a database."""
        if name == "default":
            raise ValueError("Cannot remove the default database.")
        
        if name == self.get_active_database():
            raise ValueError("Cannot remove the active database. Switch to another database first.")

        db_path = self.base_path / name
        if not db_path.exists():
            return False
            
        shutil.rmtree(db_path)
        return True

    def get_active_database(self) -> str:
        """Get the name of the currently active database."""
        if self.active_db_file.exists():
            name = self.active_db_file.read_text().strip()
            if (self.base_path / name).exists():
                return name
        return "default"

    def set_active_database(self, name: str) -> bool:
        """Set the active database."""
        if not (self.base_path / name).exists():
            return False
        self.active_db_file.write_text(name)
        return True
        
    def get_db_path(self, name: Optional[str] = None) -> Path:
        """Get the absolute path to a database."""
        if name is None:
            name = self.get_active_database()
        return self.base_path / name.strip()

# Global instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
