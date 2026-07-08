"""
universal_resolver.py — Layer 2.2 (Simplified)
================================================
GNews logic dihapus karena URLnya sudah tidak statik lagi.
Fokus: Handle URL langsung & Shortener (Bit.ly, dll).
"""
import re
import requests
import random
from dataclasses import dataclass
from typing import Optional

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]
BLOCKED_HTML_PATTERNS = [r'access denied', r'cloudflare', r'checking your browser', r'captcha', r'enable javascript']

@dataclass
class FetchResult:
    status: str
    reason: str = ""
    final_url: Optional[str] = None
    html: Optional[str] = None

def fetch_article(url: str) -> FetchResult:
    if not url:
        return FetchResult(status="dead_link", reason="empty_url")
        
    final_url = url
    
    # 1. Handle Shortener (Bit.ly, dll)
    if any(s in url for s in ["bit.ly", "tinyurl", "goo.gl", "t.co"]):
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            r = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            if r.ok and r.url and r.url != url:
                final_url = r.url
            else:
                return FetchResult(status="dead_link", reason="shortener_failed")
        except requests.exceptions.Timeout:
            return FetchResult(status="timeout", reason="shortener_timeout")
        except Exception:
            return FetchResult(status="network_error", reason="shortener_error")

    # 2. Fetch HTML dari URL asli
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "id-ID,id;q=0.9"}
        resp = requests.get(final_url, headers=headers, timeout=20, allow_redirects=True)
        
        if resp.status_code in [403, 429]:
            return FetchResult(status="blocked", reason=f"http_{resp.status_code}", final_url=final_url)
        if not resp.ok:
            return FetchResult(status="dead_link", reason=f"http_{resp.status_code}", final_url=final_url)
            
        # 3. Pre-Validation WAF
        lower_html = resp.text[:3000].lower()
        for pattern in BLOCKED_HTML_PATTERNS:
            if re.search(pattern, lower_html):
                return FetchResult(status="blocked", reason="waf_cloudflare", final_url=final_url)
                
        return FetchResult(status="ok", reason="fetch_success", final_url=final_url, html=resp.text)
        
    except requests.exceptions.Timeout:
        return FetchResult(status="timeout", reason="media_timeout", final_url=final_url)
    except Exception:
        return FetchResult(status="network_error", reason="media_error", final_url=final_url)