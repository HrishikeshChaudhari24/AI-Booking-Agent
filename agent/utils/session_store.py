import os, json, logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

try:
    import redis  # type: ignore
    from cryptography.fernet import Fernet
except ModuleNotFoundError:  # libs not installed in dev env
    redis = None  # type: ignore
    Fernet = None  # type: ignore

logger = logging.getLogger(__name__)


class _DummyStore:
    """No-op store used when REDIS_URL or cryptography is absent so code still runs."""

    def set_session(self, *_, **__):
        pass

    def get_session(self, *_):
        return {}

    def delete_session(self, *_):
        pass

    def add_user_session(self, *_, **__):
        pass

    def remove_user_session(self, *_, **__):
        pass

    def get_user_sessions(self, *_):
        return []


class RedisSessionStore:
    """Simple encrypted session store backed by Redis.

    The value is encrypted with Fernet using the key provided in
    SESSION_ENCRYPTION_KEY env-var (required).
    """

    def __init__(self):
        if redis is None or Fernet is None:
            raise RuntimeError("redis or cryptography not installed")

        redis_url = os.getenv("REDIS_URL")
        enc_key = os.getenv("SESSION_ENCRYPTION_KEY")
        if not redis_url or not enc_key:
            raise RuntimeError("REDIS_URL and SESSION_ENCRYPTION_KEY must be set")

        self.client = redis.from_url(redis_url, decode_responses=True)
        self.fernet = Fernet(enc_key.encode())

    # ---------------- core helpers ------------------

    def set_session(self, session_id: str, data: Dict[str, Any], ttl: int = 60 * 60 * 24 * 7):  # 7 days
        """Set session with longer TTL for better persistence"""
        try:
            val = self.fernet.encrypt(json.dumps(data).encode()).decode()
            self.client.setex(f"session:{session_id}", ttl, val)
            logger.info(f"Session stored successfully: {session_id}")
        except Exception as e:
            logger.error(f"Failed to store session {session_id}: {e}")
            raise

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session with better error handling"""
        try:
            raw = self.client.get(f"session:{session_id}")
            if not raw:
                logger.debug(f"Session not found: {session_id}")
                return {}
            
            decrypted_data = json.loads(self.fernet.decrypt(raw.encode()).decode())
            logger.debug(f"Session retrieved successfully: {session_id}")
            return decrypted_data
        except Exception as e:
            logger.warning(f"Failed to decrypt session {session_id}: {e}")
            self.delete_session(session_id)
            return {}


    def delete_session(self, session_id: str):
        self.client.delete(f"session:{session_id}")

    # ---------- per-user mapping helpers ------------

    def add_user_session(self, user_email: str, session_id: str, ttl: int = 60 * 60 * 24 * 30):
        self.client.setex(f"user_sessions:{user_email}:{session_id}", ttl, "1")

    def remove_user_session(self, user_email: str, session_id: str):
        self.client.delete(f"user_sessions:{user_email}:{session_id}")

    def get_user_sessions(self, user_email: str) -> List[str]:
        return [k.split(":")[-1] for k in self.client.keys(f"user_sessions:{user_email}:*")]


# Factory used by controller

def get_store():
    # --- Enhanced startup diagnostics ---
    redis_url = os.getenv("REDIS_URL")
    enc_key = os.getenv("SESSION_ENCRYPTION_KEY")
    
    logger.info("─" * 60)
    logger.info("Attempting to initialize RedisSessionStore...")
    if not redis_url:
        logger.error("  - REDIS_URL environment variable: NOT FOUND")
    else:
        # Avoid logging the full password
        safe_url = redis_url.split('@')[-1]
        logger.info(f"  - REDIS_URL found: pointing to ...@{safe_url}")

    if not enc_key:
        logger.error("  - SESSION_ENCRYPTION_KEY environment variable: NOT FOUND")
    else:
        logger.info(f"  - SESSION_ENCRYPTION_KEY found: length={len(enc_key)}")
    logger.info("─" * 60)
    # --- End diagnostics ---

    try:
        store = RedisSessionStore()
        logger.info("✅ RedisSessionStore initialized successfully.")
        return store
    except Exception as e:
        logger.error("=" * 60)
        logger.error(">>> FAILED to initialize RedisSessionStore. Using dummy fallback. <<<")
        logger.exception("  - Reason for failure:") # This will print the full traceback
        logger.error("=" * 60)

        return _DummyStore() 
