import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory credential cache that used to live in booking_agent.py.  When other
# modules need it they should import it from here so there is only one source
# of truth.
# ---------------------------------------------------------------------------
user_credentials: Dict[str, Any] = {}

# Will call init_db() after all functions are defined (bottom of file)

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def reset_database() -> None:
    """Reset the entire database – removes it and recreates the schema."""
    try:
        if os.path.exists("user_data.db"):
            os.remove("user_data.db")
            logger.info("Old database removed")

        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()

        # credentials table
        c.execute(
            """CREATE TABLE credentials (
                email TEXT PRIMARY KEY,
                token TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )

        # conversations table
        c.execute(
            """CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        
        # global appointments table
        c.execute("""
            CREATE TABLE global_appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                title TEXT DEFAULT 'Appointment',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        c.execute("""
            CREATE INDEX idx_global_appointments_date_time 
            ON global_appointments(date, start_time, end_time)
        """)

        conn.commit()
        conn.close()

        # Clear in-memory cache so it stays in sync with disk
        user_credentials.clear()
        logger.info("Fresh database created successfully!")

    except Exception as e:
        logger.exception("Database reset failed: %s", e)
        raise


def init_db() -> None:
    """Enhanced database initialization with browser sessions"""
    try:
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        
        # Existing tables...
        c.execute("""CREATE TABLE IF NOT EXISTS credentials (
            email TEXT PRIMARY KEY,
            token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            state TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # ADD THIS: Browser sessions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS browser_sessions (
                session_id TEXT PRIMARY KEY,
                user_email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add index for faster lookups
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_browser_sessions_email 
            ON browser_sessions(user_email)
        ''')
        
        # Rest of your existing code...
        conn.commit()
        conn.close()
        logger.info("Database initialization completed successfully")
        
    except Exception:
        logger.exception("Database initialization failed")
        reset_database()



# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def store_user_credentials(email: str, credentials) -> None:
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    current_time = datetime.now().isoformat()
    c.execute(
        "INSERT OR REPLACE INTO credentials (email, token, created_at) VALUES (?, ?, ?)",
        (email, credentials.to_json(), current_time),
    )
    conn.commit()
    conn.close()


def load_user_credentials(email: str):
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute("SELECT token, created_at FROM credentials WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()

    if not result:
        return None

    token, created_at_str = result
    created_at = datetime.fromisoformat(created_at_str)

    # Expire after 30 minutes
    if datetime.now() - created_at > timedelta(minutes=30):
        logger.info("Credentials expired for %s – removing from cache", email)
        delete_user_credentials(email)
        return None

    return token


def delete_user_credentials(email: str) -> None:
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute("DELETE FROM credentials WHERE email = ?", (email,))
    conn.commit()
    conn.close()

    # Remove in-memory copy
    user_credentials.pop(email, None)


def cleanup_expired_credentials() -> None:
    """Remove credentials older than 30 minutes from both disk and memory."""
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    try:
        c.execute("SELECT created_at FROM credentials LIMIT 1")  # quick existence check
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            logger.warning("Credentials table missing during cleanup – reinitialising DB")
            init_db()
            conn.close()
            return

    expiry_time = (datetime.now() - timedelta(minutes=30)).isoformat()
    c.execute("SELECT email FROM credentials WHERE created_at < ?", (expiry_time,))
    expired = [row[0] for row in c.fetchall()]

    if expired:
        logger.info("Cleaning up expired credentials: %s", expired)
        c.execute("DELETE FROM credentials WHERE created_at < ?", (expiry_time,))
        conn.commit()
        for email in expired:
            user_credentials.pop(email, None)

    conn.close()


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def save_conversation_state(conversation_id: str, state: Dict[str, Any]) -> None:
    try:
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        c.execute(
            """INSERT OR REPLACE INTO conversations (id, state, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)""",
            (conversation_id, json.dumps(state)),
        )
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        if "no such table: conversations" in str(e):
            logger.warning("Conversations table missing – re-initialising DB")
            init_db()
            save_conversation_state(conversation_id, state)  # retry
        else:
            logger.exception("Error saving conversation state")
            raise


def load_conversation_state(conversation_id: str) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        c.execute("SELECT state FROM conversations WHERE id = ?", (conversation_id,))
        row = c.fetchone()
        conn.close()
        return json.loads(row[0]) if row else {}
    except sqlite3.OperationalError as e:
        if "no such table: conversations" in str(e):
            logger.warning("Conversations table missing – re-initialising DB")
            init_db()
            return {}
        logger.exception("Error loading conversation state")
        return {}
    except Exception:
        logger.exception("Unexpected error loading conversation state")
        return {}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_database_health() -> bool:
    """Return True if required tables/columns exist and are correct."""
    try:
        conn = sqlite3.connect("user_data.db")
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in c.fetchall()]

        required = ["credentials", "conversations"]
        if any(t not in tables for t in required):
            logger.warning("Missing tables: %s", set(required) - set(tables))
            return False

        for tbl in required:
            c.execute(f"PRAGMA table_info({tbl})")
            cols = [col[1] for col in c.fetchall()]
            logger.info("%s columns: %s", tbl, cols)

        conn.close()
        return True
    except Exception:
        logger.exception("Database health check failed")
        return False

# ---------------------------------------------------------------------------
# Run DB initialisation when module is imported
# ---------------------------------------------------------------------------

try:
    init_db()
except Exception:
    logger.exception("Database init failed during import") 
