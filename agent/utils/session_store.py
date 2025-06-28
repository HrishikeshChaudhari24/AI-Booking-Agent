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

    def set_session(self, session_id: str, data: Dict[str, Any], ttl: int = 60 * 60 * 24 * 30):
        val = self.fernet.encrypt(json.dumps(data).encode()).decode()
        self.client.setex(f"session:{session_id}", ttl, val)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        raw = self.client.get(f"session:{session_id}")
        if not raw:
            return {}
        try:
            return json.loads(self.fernet.decrypt(raw.encode()).decode())
        except Exception as e:
            logger.warning("Failed to decrypt session %s: %s", session_id, e)
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
    try:
        return RedisSessionStore()
    except Exception as e:
        logger.warning("Redis session store not available – falling back to dummy: %s", e)
        return _DummyStore() 