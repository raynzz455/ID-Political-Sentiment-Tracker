"""
gnews_resolver.py — Local Heavy GNews Resolver (Refactored v10 - Hybrid Optimized)
=============================================================================
Arsitektur Hybrid (Sesuai Review Engineering):
  1. ASYNC BASE: Menggunakan asyncio untuk throughput tinggi (I/O Bound).
  2. RESOURCE EFFICIENT: 1 Browser, 1 Context, N Pages. (Hemat RAM).
  3. CONFIGURABLE: Semua parameter (concurrency, timeout, visible) via CLI.
  4. AUDITABLE LOGGING: Mencatat Original URL -> Resolved URL -> Domain -> Text Length.
  5. COMPREHENSIVE STATS: Statistik akhir mencakup Success Rate, Avg Time, dan Kategori Gagal.
"""
import os
import sys
import time
import asyncio
import argparse
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).resolve().parents[2]))
from devtools.common import get_supabase, build_text_hash

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
    from trafilatura import extract as traf_extract
except ImportError as e:
    print(f"[ERROR] {e}\nPastikan: pip install playwright trafilatura")
    print("Dan jalankan: playwright install chromium")
    sys.exit(1)

from packages.shared import constants as pc

def extract_full_text(html: str) -> str:
    if not html: return ""
    return traf_extract(html, include_comments=False, include_tables=False) or ""

async def process_url(context, art: dict, semaphore: asyncio.Semaphore, timeout: int, stats: Counter) -> dict:
    """Memproses 1 URL secara asynchronous menggunakan Page dari Context bersama."""
    gnews_url = art["source_url"]
    current_attempts = art.get("recovery_attempts", 0) + 1
    start_time = time.perf_counter()
    
    if len(gnews_url) < 80:
        stats["invalid_url"] += 1
        return {
            "id": art["id"],
            "recovery_attempts": pc.MAX_RECOVERY_RETRY,
            "recovery_status": "failed_truncated_url"
        }
        
    async with semaphore:
        page = await context.new_page()
        resolved_url = None
        html_content = None
        
        try:
            await page.goto(gnews_url, wait_until="domcontentloaded", timeout=timeout * 1000)
            
            try:
                await page.wait_for_url(lambda url: "google.com" not in url, timeout=timeout * 1000)
            except PlaywrightTimeoutError:
                stats["timeout"] += 1
                pass
            
            final_url = page.url
            if "google.com" not in final_url and "gstatic.com" not in final_url:
                resolved_url = final_url
                html_content = await page.content()
            else:
                stats["google_loop"] += 1
                
        except PlaywrightError:
            stats["http_error"] += 1
        except Exception:
            stats["http_error"] += 1
        finally:
            await page.close()
            
    if resolved_url and html_content:
        full_text = extract_full_text(html_content)
        
        if full_text and len(full_text) >= 200:
            domain = urlparse(resolved_url).netloc.replace("www.", "")
            elapsed = time.perf_counter() - start_time
            stats["resolved"] += 1
            stats["total_time"] += elapsed
            
            # Detailed Logging
            print(f"\n  ✅ ID: {art['id'][:8]} | {elapsed:.1f}s")
            print(f"     Original : {gnews_url[:70]}...")
            print(f"     Resolved : {resolved_url[:70]}")
            print(f"     Domain   : {domain}")
            print(f"     Text Len : {len(full_text)} chars")
            
            current_meta = dict(art.get("metadata") or {})
            current_meta["is_snippet"] = False
            current_meta["resolved_url"] = resolved_url
            current_meta["resolver_method"] = "playwright_hybrid_v10"
            
            return {
                "id": art["id"],
                "text": full_text,
                "status": pc.STATUS_ENRICHED,
                "content_type": "FULLTEXT",
                "metadata": current_meta,
                "recovery_attempts": current_attempts,
                "recovery_status": pc.RECOVERY_RESOLVED,
                "content_hash": build_text_hash(full_text)
            }
        else:
            stats["text_too_short"] += 1
            
    if not resolved_url:
        print(f"  ❌ ID: {art['id'][:8]} | Failed to resolve", end="")
        if not any(k in str(stats) for k in ["timeout", "google_loop", "http_error"]):
            stats["redirect_failed"] += 1
            
    return {
        "id": art["id"],
        "recovery_attempts": current_attempts,
        "recovery_status": pc.RECOVERY_FAILED
    }

async def async_main(sb, args):
    total_processed = 0
    batch_num = 1
    global_start = time.perf_counter()
    stats = Counter()
    
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"[PLAYWRIGHT] Mode: {'Unlimited' if args.max_total == 0 else f'Max {args.max_total}'} | Batch: {args.limit} | Concurrency: {args.concurrency} | Timeout: {args.timeout}s | Visible: {args.visible}")
    
    async with async_playwright() as p:
        # 1 Browser, 1 Context
        browser = await p.chromium.launch(
            headless=not args.visible,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="id-ID"
        )
        
        while True:
            if args.max_total > 0 and total_processed >= args.max_total:
                break
                
            print(f"\n--- Batch {batch_num} ---")
            res = sb.table("raw_texts") \
                .select("id, source_url, metadata, recovery_attempts") \
                .eq("status", pc.STATUS_FAILED) \
                .eq("content_type", "SNIPPET") \
                .lt("recovery_attempts", pc.MAX_RECOVERY_RETRY) \
                .limit(args.limit) \
                .execute()
                    
            articles = res.data or []
            if not articles:
                print("[PLAYWRIGHT] Tidak ada lagi artikel untuk di-resolve.")
                break
                
            updates = []
            
            # N Pages from 1 Context
            tasks = [process_url(context, art, semaphore, args.timeout, stats) for art in articles]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    stats["crash"] += 1
                    print(f"  💥 Task Error: {str(result)[:50]}")
                else:
                    updates.append(result)
                        
            if updates:
                try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
                except Exception as e: print(f"[DB_ERROR] {e}")
                
            total_processed += len(articles)
            batch_num += 1
            
        await context.close()
        await browser.close()
        
    elapsed = time.perf_counter() - global_start
    
    # Comprehensive Statistics
    total_resolved = stats["resolved"]
    print(f"\n{'='*60}")
    print(f"📊 RECOVERY STATISTICS REPORT")
    print(f"{'='*60}")
    print(f"  Total Processed    : {total_processed}")
    print(f"  Total Resolved    : {total_resolved}")
    print(f"  Success Rate      : {(total_resolved/total_processed*100):.1f}%" if total_processed > 0 else "0.0%")
    print(f"  Average Resolve   : {(stats['total_time']/total_resolved):.2f}s" if total_resolved > 0 else "0.00s")
    print(f"  Total Exec Time   : {elapsed:.2f}s ({elapsed/60:.1f}m)")
    print(f"{'─'*60}")
    print(f"  Failure Breakdown :")
    print(f"    - Timeout       : {stats['timeout']}")
    print(f"    - Redirect Fail : {stats['redirect_failed']}")
    print(f"    - HTTP Error    : {stats['http_error']}")
    print(f"    - Google Loop   : {stats['google_loop']}")
    print(f"    - Text Too Short: {stats['text_too_short']}")
    print(f"    - Invalid URL   : {stats['invalid_url']}")
    print(f"    - Task Crash    : {stats['crash']}")
    print(f"{'='*60}\n")

def main(args) -> None:
    sb = get_supabase()
    asyncio.run(async_main(sb, args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local GNews Playwright Resolver (Hybrid)")
    parser.add_argument("--limit", type=int, default=20, help="Jumlah row per batch (default 20)")
    parser.add_argument("--max-total", type=int, default=0, help="Batas total proses (0 = unlimited)")
    parser.add_argument("--concurrency", type=int, default=3, help="Jumlah tab diproses bersamaan (default 3)")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout tunggu redirect dalam detik (default 15)")
    parser.add_argument("--visible", action="store_true", help="Tampilkan browser di layar (headless=False)")
    parser.add_argument("--url", type=str, help="Test resolve 1 URL GNews secara manual")
    args = parser.parse_args()
    
    if args.url:
        print("[PLAYWRIGHT] Mode Testing Manual (1 URL)")
        async def test_url():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False, args=['--disable-blink-features=AutomationControlled'])
                context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                page = await context.new_page()
                try:
                    print(f"  -> Membuka: {args.url}")
                    await page.goto(args.url, wait_until="domcontentloaded", timeout=20000)
                    
                    print("  Menunggu redirect Google (max 15s)...", end=" ")
                    resolved = None
                    try:
                        await page.wait_for_url(lambda url: "google.com" not in url, timeout=15000)
                        resolved = page.url
                    except:
                        pass
                                
                    if resolved:
                        print(f"\n  Hasil URL Final: {resolved}")
                        html = await page.content()
                        text = traf_extract(html, include_comments=False, include_tables=False) or ""
                        print(f"  Panjang Teks: {len(text)} karakter")
                        print(f"  Preview: {text[:200]}...")
                    else:
                        print("❌ Gagal keluar dari Google (Timeout 15s)")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                finally:
                    await browser.close()
        asyncio.run(test_url())
    else:
        main(args)