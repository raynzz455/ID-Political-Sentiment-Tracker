"""
db_client.py - Database Connection Manager
"""
import os
from supabase import create_client, Client

def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise Exception("Supabase URL/Key not set in environment.")
    return create_client(url, key)