"""
colab_keepalive.py
==================
Anti-disconnect script untuk Colab.

Colab free tier disconnect setelah ~90 menit idle. Script ini:
1. Simulate keyboard activity (click pada cell)
2. Print heartbeat setiap 60 detik
3. Save progress marker ke Drive

Usage di Colab:
  # Jalankan di cell terpisah (background)
  import threading
  from colab_keepalive import keepalive_loop
  t = threading.Thread(target=keepalive_loop, daemon=True)
  t.start()
  print("Keep-alive started")

  # Atau run langsung (blocking):
  !python colab_keepalive.py
"""
import time
import threading
from datetime import datetime
from pathlib import Path


def keepalive_loop(interval_sec: int = 60):
    """Loop yang print heartbeat setiap interval untuk mencegah Colab disconnect.

    Args:
        interval_sec: interval heartbeat (default 60s — Colab timeout ~90s)
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Keep-alive started (interval={interval_sec}s)")
    start = time.time()

    while True:
        try:
            elapsed = time.time() - start
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            # Heartbeat message
            msg = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                   f"♥ Alive — runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")

            # Try to write marker to Drive (if mounted)
            drive_marker = Path("/content/drive/MyDrive/finetuning_progress")
            if drive_marker.exists():
                marker_file = drive_marker / "keepalive.txt"
                marker_file.write_text(
                    f"{msg}\n"
                    f"Last update: {datetime.now().isoformat()}\n"
                )

            print(msg, flush=True)
            time.sleep(interval_sec)

        except KeyboardInterrupt:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Keep-alive stopped by user")
            break
        except Exception as e:
            # Don't crash on error — just log and continue
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Keep-alive error: {e}", flush=True)
            time.sleep(interval_sec)


def click_simulation():
    """Simulate mouse click pada Colab cell (anti-idle).

    Uses JavaScript to dispatch click event. Only works in Colab frontend.
    """
    try:
        from google.colab import output  # type: ignore
        output.eval_js('''
            function ClickConnect() {
                var btn = document.querySelector("colab-connect-button");
                if (btn) {
                    btn.click();
                }
            }
            setInterval(ClickConnect, 60000);
        ''')
        print("Click simulation started (60s interval)")
    except ImportError:
        print("Not in Colab — click simulation skipped")
    except Exception as e:
        print(f"Click simulation failed: {e}")


def start_keepalive(interval_sec: int = 60, with_clicks: bool = True):
    """Start keep-alive in background thread.

    Usage:
        from colab_keepalive import start_keepalive
        start_keepalive()
        # Continue with other work...
    """
    print(f"Starting keep-alive (interval={interval_sec}s, clicks={with_clicks})...")

    # Start heartbeat thread
    t = threading.Thread(target=keepalive_loop, args=(interval_sec,), daemon=True)
    t.start()

    # Start click simulation (Colab only)
    if with_clicks:
        try:
            click_simulation()
        except Exception:
            pass

    print("✅ Keep-alive running in background")
    print("   Heartbeat every 60s + click simulation")
    print("   To stop: restart kernel or interrupt")


if __name__ == "__main__":
    print("=" * 60)
    print("  Colab Keep-Alive — Anti-Disconnect")
    print("=" * 60)
    print()
    print("This script runs forever. Press Ctrl+C to stop.")
    print("Heartbeat every 60s prevents Colab idle disconnect.")
    print()
    start_keepalive()
