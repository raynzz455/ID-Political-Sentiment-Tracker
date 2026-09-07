"""
db_client.py - Database Connection Manager
"""
import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise Exception("Supabase URL/Key not set in environment. Pastikan file .env ada di root folder.")
    return create_client(url, key)


# FIX SF#5 (MEDIUM): Shared bulk_update helper with retry logic.
# Before: 5 workers had pattern `try: sb.rpc("bulk_update_raw_texts", ...) except: log`
# — no retry, transient failures lose all updates.
# After: Shared helper with 3 retries + exponential backoff.
def bulk_update_with_retry(sb: Client, updates: list, chunk_size: int = 50,
                            max_retries: int = 3) -> tuple[int, int]:
    """Bulk update raw_texts with retry logic.

    Args:
        sb: Supabase client
        updates: list of update dicts (each must have "id" key)
        chunk_size: RPC chunk size (default 50)
        max_retries: max retry attempts per chunk

    Returns:
        (succeeded_count, failed_count)
    """
    succeeded = 0
    failed = 0
    for i in range(0, len(updates), chunk_size):
        chunk = updates[i:i + chunk_size]
        for attempt in range(max_retries):
            try:
                sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
                succeeded += len(chunk)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"bulk_update failed (chunk {i//chunk_size + 1}) "
                        f"after {max_retries} retries: {e}"
                    )
                    failed += len(chunk)
                else:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"bulk_update retry {attempt + 1}/{max_retries} "
                        f"(chunk {i//chunk_size + 1}): {e} — waiting {wait}s"
                    )
                    time.sleep(wait)
    return succeeded, failed