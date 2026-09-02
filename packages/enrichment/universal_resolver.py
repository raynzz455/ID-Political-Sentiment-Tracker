"""
universal_resolver.py v3 — True Universal Resolver & Strategy Pattern
=======================================================================
Evolusi dari "Universal Fetcher" menjadi "Universal Resolver".
"""

import random
import re
import base64
import threading
import time
from typing import Optional
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import requests

# UBAH IMPORT INI KE PACKAGES.SHARED
from packages.shared.constants import (
    FETCH_OK, FETCH_BLOCKED, FETCH_TIMEOUT, FETCH_DEAD_LINK, FETCH_NETWORK_ERROR,
    REASON_EMPTY_URL, REASON_SHORTENER_FAILED, REASON_SHORTENER_TIMEOUT,
    REASON_SHORTENER_ERROR, REASON_WAF_BLOCKED, REASON_FETCH_SUCCESS,
    REASON_MEDIA_TIMEOUT, REASON_MEDIA_ERROR, REASON_GNEWS_SNIPPET_ONLY,
    http_reason,
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

SHORTENER_DOMAINS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "feedburner"]
GNEWS_DOMAIN = "news.google.com"

MAX_CONCURRENT_PER_DOMAIN = 2
MEDIA_FETCH_MAX_ATTEMPTS = 3
MEDIA_FETCH_BACKOFF_SECONDS = [1, 2, 4] 

BLOCKED_HTML_PATTERNS = [
    r"just a moment", r"checking your browser before accessing",
    r"attention required.{0,10}cloudflare", r"verify you are human",
    r"ddos protection by cloudflare", r"enable javascript and cookies to continue",
    r"unusual traffic from your computer network", r"sedang memeriksa browser anda",
]

_thread_local = threading.local()
_domain_semaphores: dict[str, threading.Semaphore] = {}
_domain_semaphore_lock = threading.Lock()


@dataclass
class FetchResult:
    """Kontrak kaya antara resolver & pemanggilnya (Layer 2.5+)."""
    status: str
    reason: str = ""
    original_url: Optional[str] = None
    resolved_url: Optional[str] = None
    canonical_url: Optional[str] = None
    html: Optional[str] = None
    redirect_count: int = 0
    resolver_method: str = "unknown"
    confidence: float = 0.0
    fetch_metadata: dict = field(default_factory=dict)


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session

def _get_domain_semaphore(url: str) -> threading.Semaphore:
    domain = urlparse(url).netloc
    sem = _domain_semaphores.get(domain)
    if sem is None:
        with _domain_semaphore_lock:
            sem = _domain_semaphores.setdefault(domain, threading.Semaphore(MAX_CONCURRENT_PER_DOMAIN))
    return sem

def _get_with_retry(session: requests.Session, url: str, headers: dict, timeout: int) -> requests.Response:
    last_exc: Exception = requests.exceptions.Timeout("no attempt made")
    for attempt in range(MEDIA_FETCH_MAX_ATTEMPTS):
        try:
            return session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < MEDIA_FETCH_MAX_ATTEMPTS - 1:
                time.sleep(MEDIA_FETCH_BACKOFF_SECONDS[attempt])
    raise last_exc

def _is_interstitial(html: str) -> bool:
    if not html: return False
    sample = html[:3000].lower()
    return any(re.search(p, sample) for p in BLOCKED_HTML_PATTERNS)

def _extract_canonical(html: str) -> Optional[str]:
    """Mencari tag <link rel="canonical"> di HTML untuk normalisasi URL."""
    if not html: return None    
    match = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        return match.group(1)        
    match = re.search(r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']canonical["\']', html, re.IGNORECASE)
    if match:
        return match.group(1)
        
    return None

# ─────────────────────────────────────────────────────────────
# RESOLVER STRATEGIES
# ─────────────────────────────────────────────────────────────

def _resolve_gnews(url: str) -> FetchResult:
    return FetchResult(
        status=FETCH_OK, reason=REASON_GNEWS_SNIPPET_ONLY,
        original_url=url, resolved_url=url,
        resolver_method="gnews_snippet_fallback", confidence=0.50
    )

def _resolve_shortener(url: str, headers: dict, session: requests.Session) -> tuple[str, Optional[FetchResult]]:
    if not any(s in url for s in SHORTENER_DOMAINS):
        return url, None
    try:
        r = session.head(url, headers=headers, timeout=10, allow_redirects=True)
        if r.ok and r.url and r.url != url:
            return r.url, None
        return url, FetchResult(status=FETCH_DEAD_LINK, reason=REASON_SHORTENER_FAILED, original_url=url)
    except requests.exceptions.Timeout:
        return url, FetchResult(status=FETCH_TIMEOUT, reason=REASON_SHORTENER_TIMEOUT, original_url=url)
    except Exception:
        return url, FetchResult(status=FETCH_NETWORK_ERROR, reason=REASON_SHORTENER_ERROR, original_url=url)

# ─────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def fetch_article(url: str, metadata: Optional[dict] = None) -> FetchResult:
    if not url:
        return FetchResult(status=FETCH_DEAD_LINK, reason=REASON_EMPTY_URL)
    if metadata and metadata.get("resolved_url"):
        url = metadata["resolved_url"]
    elif GNEWS_DOMAIN in url:
        gnews_result = _resolve_gnews(url)
        if gnews_result.reason != REASON_GNEWS_SNIPPET_ONLY:
            url = gnews_result.resolved_url
        else:
            return gnews_result

    headers = {"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "id-ID,id;q=0.9,en;q=0.8"}
    session = _get_session()

    final_url, shortener_error = _resolve_shortener(url, headers, session)
    if shortener_error:
        return shortener_error

    try:
        with _get_domain_semaphore(final_url):
            resp = _get_with_retry(session, final_url, headers, timeout=20)

        if resp.status_code in (403, 429):
            return FetchResult(status=FETCH_BLOCKED, reason=http_reason(resp.status_code), original_url=url, resolved_url=final_url)
        if not resp.ok:
            return FetchResult(status=FETCH_DEAD_LINK, reason=http_reason(resp.status_code), original_url=url, resolved_url=final_url)

        if _is_interstitial(resp.text):
            return FetchResult(status=FETCH_BLOCKED, reason=REASON_WAF_BLOCKED, original_url=url, resolved_url=final_url)

        canonical = _extract_canonical(resp.text)
        resolver_method = "direct_get"
        confidence = 0.90
        
        final_resolved_url = canonical if canonical and canonical != final_url else final_url
        if canonical and canonical != final_url:
            resolver_method = "canonical_resolved"
            confidence = 0.95

        resolved_domain = urlparse(final_resolved_url).netloc.replace("www.", "")

        return FetchResult(
            status=FETCH_OK, reason=REASON_FETCH_SUCCESS,
            original_url=url, resolved_url=final_resolved_url, canonical_url=canonical,
            html=resp.text, redirect_count=len(resp.history),
            resolver_method=resolver_method, confidence=confidence,
            fetch_metadata={"content_type": resp.headers.get("Content-Type", ""), "resolved_domain": resolved_domain}
        )

    except requests.exceptions.Timeout:
        return FetchResult(status=FETCH_TIMEOUT, reason=REASON_MEDIA_TIMEOUT, original_url=url, resolved_url=final_url)
    except Exception:
        return FetchResult(status=FETCH_NETWORK_ERROR, reason=REASON_MEDIA_ERROR, original_url=url, resolved_url=final_url)