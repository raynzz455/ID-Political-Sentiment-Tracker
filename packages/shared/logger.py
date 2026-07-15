"""
logger.py - Pipeline Observability
"""
import os
from datetime import datetime, timezone
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def _get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        raise Exception("Supabase URL/Key not set in environment.")
    return create_client(SUPABASE_URL, SERVICE_KEY)

def start_run(worker_name: str, version: str = "1.0") -> str | None:
    sb = _get_client()
    res = sb.table("pipeline_runs").insert({
        "worker_name": worker_name,
        "version": version,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat()
    }).execute()
    return res.data[0]["id"] if res.data else None

def finish_run(run_id: str, processed: int, succeeded: int, failed: int):
    if not run_id: return
    sb = _get_client()
    now = datetime.now(timezone.utc)
    res = sb.table("pipeline_runs").select("started_at").eq("id", run_id).execute()
    if not res.data: return
    
    start_time = datetime.fromisoformat(res.data[0]["started_at"])
    duration = (now - start_time).total_seconds()
    
    sb.table("pipeline_runs").update({
        "finished_at": now.isoformat(),
        "duration_seconds": duration,
        "articles_processed": processed,
        "articles_succeeded": succeeded,
        "articles_failed": failed,
        "status": "completed"
    }).eq("id", run_id).execute()