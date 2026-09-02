"""
common.py — Shared Utilities for DevTools
==========================================
Menghilangkan boilerplate repetisi (load_dotenv, get_client, dll) di seluruh devtools.
"""
import os
import sys
import hashlib
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

def get_supabase() -> Client:
    """Inisialisasi Supabase Client."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("[ERROR] Set SUPABASE_URL & SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(url, key)

def build_text_hash(text: str) -> str:
    """Generate SHA256 hash untuk teks."""
    return hashlib.sha256(text.encode()).hexdigest()

def setup_argparse(description: str) -> argparse.ArgumentParser:
    """Setup argparser standar untuk devtools."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, default=20, help="Jumlah row yang diproses (default 20)")
    return parser