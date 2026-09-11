"""
maintenance_worker.py — Weekly Database Cleanup & Maintenance
==============================================================
Menjalankan pembersihan database berkala untuk menjaga Supabase
tetap di bawah limit 500MB free-tier.

APA YANG DILAKUKAN:
  1. Snapshot ukuran DB SEBELUM cleanup (via get_db_size_report RPC)
  2. Heavy dedup + orphan cleanup (via run_weekly_cleanup RPC)
  3. Drop old partitions aggressive 4 bulan (via drop_old_partitions_aggressive)
  4. VACUUM ANALYZE (opsional, butuh DATABASE_URL = koneksi langsung)
  5. Snapshot ukuran DB SESUDAH cleanup + laporan

CARA MENJALANKAN:
  # Manual (lokal):
  python packages/db/maintenance_worker.py

  # Via GitHub Action (weekly):
  # Lihat .github/workflows/db-maintenance.yml

ENV VARS:
  WAJIB:
    SUPABASE_URL              — project URL
    SUPABASE_SERVICE_ROLE_KEY — service role key (bypass RLS)
  OPSIONAL (untuk VACUUM):
    DATABASE_URL              — direct Postgres connection string
                                (Supabase Settings → Database → Connection string)
                                Tanpa ini, VACUUM di-skip dengan warning.

CATATAN:
  - VACUUM tidak bisa dijalankan via REST API / RPC (PostgREST).
    Butuh koneksi langsung (psycopg2) atau Supabase SQL Editor.
  - pg_cron sudah handle light cleanup tiap 6 jam (lihat seed 17).
    Worker ini fokus pada heavy dedup mingguan + VACUUM.
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_supabase_client():
    """Get Supabase REST client."""
    from packages.shared.db_client import get_client
    return get_client()


def pretty_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def report_before(sb) -> dict:
    """Snapshot DB size before cleanup."""
    logger.info("=" * 60)
    logger.info("STEP 1: Snapshot ukuran DB SEBELUM cleanup")
    logger.info("=" * 60)
    try:
        res = sb.rpc("get_db_size_report").execute()
        data = res.data
        logger.info(f"  DB size: {data.get('db_size_pretty')} "
                    f"({data.get('db_size_bytes'):,} bytes)")
        tables = data.get("tables", {})
        for name, count in tables.items():
            logger.info(f"  {name}: {count:,} rows")
        top = data.get("top_tables_by_size", [])
        if top:
            logger.info("  --- Top tables by size ---")
            for t in top[:5]:
                logger.info(f"    {t['table']}: {t['size_pretty']}")
        return data
    except Exception as e:
        logger.error(f"  Gagal get_db_size_report: {e}")
        logger.error("  Pastikan RPC functions sudah di-install "
                     "(jalankan seed 18_cleanup_rpc_functions.sql)")
        return {}


def run_weekly_cleanup(sb) -> dict:
    """Heavy dedup + orphan cleanup via RPC."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Heavy dedup + orphan cleanup (run_weekly_cleanup)")
    logger.info("=" * 60)
    try:
        res = sb.rpc("run_weekly_cleanup").execute()
        data = res.data
        if not data:
            logger.warning("  RPC returned empty result")
            return {}
        logger.info(f"  Duplicates deleted:        {data.get('dups_deleted', 0):,}")
        logger.info(f"  Orphan mentions deleted:   {data.get('orphan_mentions', 0):,}")
        logger.info(f"  Orphan contexts deleted:   {data.get('orphan_contexts', 0):,}")
        logger.info(f"  Orphan map deleted:        {data.get('orphan_map', 0):,}")
        logger.info(f"  Orphan scores deleted:     {data.get('orphan_scores', 0):,}")
        logger.info(f"  Orphan highlights deleted: {data.get('orphan_highlights', 0):,}")
        logger.info(f"  Hashes deleted:            {data.get('hashes_deleted', 0):,}")
        logger.info(f"  Pipeline runs deleted:     {data.get('runs_deleted', 0):,}")
        logger.info(f"  Candidates deleted:        {data.get('candidates_deleted', 0):,}")
        return data
    except Exception as e:
        logger.error(f"  Gagal run_weekly_cleanup: {e}")
        logger.error("  Pastikan RPC function ter-install (seed 18)")
        return {}


def drop_old_partitions(sb, months: int = 4) -> dict:
    """Drop old partitions (aggressive 4-month retention)."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 3: Drop old partitions (keep {months} months)")
    logger.info("=" * 60)
    try:
        res = sb.rpc("drop_old_partitions_aggressive",
                     {"p_months": months}).execute()
        data = res.data or {}
        logger.info(f"  Done: {data.get('action', 'drop_old_partitions')}")
        return data
    except Exception as e:
        logger.error(f"  Gagal drop_old_partitions: {e}")
        return {}


def run_vacuum():
    """VACUUM ANALYZE via direct psycopg2 connection (optional).

    Butuh DATABASE_URL env var (Supabase connection string).
    Jika tidak ada, skip dengan warning.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: VACUUM ANALYZE (reclaim physical space)")
    logger.info("=" * 60)
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        logger.warning("  ⚠️  DATABASE_URL tidak di-set — VACUUM di-skip.")
        logger.warning("      VACUUM tidak bisa via REST API. Untuk reclaim")
        logger.warning("      space fisik, jalankan manual di Supabase SQL Editor:")
        logger.warning("        VACUUM (ANALYZE) raw_texts;")
        logger.warning("        VACUUM (ANALYZE) sentiment_scores;")
        logger.warning("        VACUUM (ANALYZE) entity_mentions, entity_contexts,")
        logger.warning("                     article_entity_map, raw_text_hashes;")
        logger.warning("      Atau set DATABASE_URL secret di GitHub Actions.")
        return False

    try:
        import psycopg2  # type: ignore
    except ImportError:
        logger.warning("  ⚠️  psycopg2 belum ter-install. Install dengan:")
        logger.warning("      pip install psycopg2-binary")
        logger.warning("      Lalu set DATABASE_URL. Skip VACUUM untuk sekarang.")
        return False

    tables = [
        "raw_texts", "sentiment_scores", "entity_mentions",
        "entity_contexts", "article_entity_map", "raw_text_hashes",
        "entity_highlights", "pipeline_runs", "entity_candidates",
    ]
    try:
        # autocommit WAJIB untuk VACUUM (tidak boleh dalam transaction)
        conn = psycopg2.connect(db_url, autocommit=True)
        cur = conn.cursor()
        for t in tables:
            logger.info(f"  VACUUM ANALYZE {t} ...")
            try:
                cur.execute(f"VACUUM (ANALYZE) {t};")
                logger.info(f"    ✓ done")
            except Exception as e:
                logger.warning(f"    ⚠️  {t}: {e}")
        cur.close()
        conn.close()
        logger.info("  VACUUM ANALYZE selesai ✓")
        return True
    except Exception as e:
        logger.error(f"  Gagal VACUUM: {e}")
        return False


def report_after(sb, before: dict):
    """Snapshot DB size after cleanup + delta."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Snapshot ukuran DB SESUDAH cleanup")
    logger.info("=" * 60)
    try:
        res = sb.rpc("get_db_size_report").execute()
        data = res.data
        after_bytes = data.get("db_size_bytes", 0)
        before_bytes = before.get("db_size_bytes", 0)
        delta = before_bytes - after_bytes
        logger.info(f"  DB size BEFORE: {pretty_bytes(before_bytes)}")
        logger.info(f"  DB size AFTER:  {pretty_bytes(after_bytes)} "
                    f"({data.get('db_size_pretty')})")
        if delta > 0:
            logger.info(f"  ✅ Space reclaimed (pre-VACUUM): {pretty_bytes(delta)}")
        elif delta == 0:
            logger.info(f"  ➖ No change (mungkin perlu VACUUM FULL)")
        else:
            logger.info(f"  ⚠️  DB grew by {pretty_bytes(abs(delta))} "
                        f"(ingestion baru lebih cepat dari cleanup)")
        tables = data.get("tables", {})
        before_tables = before.get("tables", {})
        logger.info("  --- Row count delta ---")
        for name, count in tables.items():
            bcount = before_tables.get(name, 0)
            d = count - bcount
            sign = "+" if d >= 0 else ""
            logger.info(f"    {name}: {bcount:,} → {count:,} ({sign}{d:,})")
        return data
    except Exception as e:
        logger.error(f"  Gagal get_db_size_report (after): {e}")
        return {}


def main():
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  DATABASE MAINTENANCE WORKER — Weekly Cleanup         ║")
    logger.info("║  Target: keep Supabase < 500MB free-tier limit        ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"  Started: {datetime.now(timezone.utc).isoformat()}")

    sb = get_supabase_client()

    before = report_before(sb)
    cleanup_result = run_weekly_cleanup(sb)
    drop_old_partitions(sb, months=4)
    run_vacuum()
    report_after(sb, before)

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ MAINTENANCE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Finished: {datetime.now(timezone.utc).isoformat()}")

    # Exit code: 0 jika cleanup jalan, 1 jika RPC gagal
    if not cleanup_result:
        logger.error("  ⚠️  Cleanup RPC gagal — perlu investigasi")
        sys.exit(1)
    logger.info("  All steps done. Monitor via db_cleanup_log table.")


if __name__ == "__main__":
    main()
