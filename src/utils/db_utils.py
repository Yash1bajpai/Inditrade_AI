import os
import sqlite3
from datetime import datetime

def connect_to_graph_db(db_path=".agents/graph_memory.sqlite"):
    """Connect to the SQLite graph memory database."""
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return None, None
        
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    now_iso = datetime.now().isoformat() + "Z"
    return db, cursor, now_iso
