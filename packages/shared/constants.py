"""
constants.py — Kamus Bersama Lintas Layer
==============================================================
SATU sumber kebenaran untuk semua string status/reason yang dipakai lintas
universal_resolver.py, enricher_worker.py, & validation_worker.py.
"""
from enum import Enum


class _CleanStrEnum(str, Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)


class RawTextStatus(_CleanStrEnum):
    PENDING = "pending"
    ENRICHED = "enriched"
    VALIDATED = "validated"
    FAILED = "failed"
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    SKIPPED = "skipped"


class FetchStatus(_CleanStrEnum):
    OK = "ok"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    DEAD_LINK = "dead_link"
    NETWORK_ERROR = "network_error"


class FetchReason(_CleanStrEnum):
    EMPTY_URL = "empty_url"
    SHORTENER_FAILED = "shortener_failed"
    SHORTENER_TIMEOUT = "shortener_timeout"
    SHORTENER_ERROR = "shortener_error"
    WAF_BLOCKED = "waf_cloudflare"
    FETCH_SUCCESS = "fetch_success"
    MEDIA_TIMEOUT = "media_timeout"
    MEDIA_ERROR = "media_error"
    GNEWS_SNIPPET_ONLY = "gnews_snippet_only"
    RSS_FULL_TEXT = "rss_full_text"
    EXTRACT_EMPTY = "extract_empty"
    THREAD_CRASH = "thread_crash"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"


def http_reason(status_code: int) -> str:
    return f"http_{status_code}"


RETRYABLE_FETCH_STATUSES = {FetchStatus.BLOCKED, FetchStatus.TIMEOUT, FetchStatus.NETWORK_ERROR}
MAX_ENRICH_RETRIES = 3

# ── Alias flat (backward-compat) ──
STATUS_PENDING = RawTextStatus.PENDING
STATUS_ENRICHED = RawTextStatus.ENRICHED
STATUS_VALIDATED = RawTextStatus.VALIDATED
STATUS_FAILED = RawTextStatus.FAILED
STATUS_QUEUED = RawTextStatus.QUEUED
STATUS_PROCESSING = RawTextStatus.PROCESSING
STATUS_PROCESSED = RawTextStatus.PROCESSED
STATUS_SKIPPED = RawTextStatus.SKIPPED

FETCH_OK = FetchStatus.OK
FETCH_BLOCKED = FetchStatus.BLOCKED
FETCH_TIMEOUT = FetchStatus.TIMEOUT
FETCH_DEAD_LINK = FetchStatus.DEAD_LINK
FETCH_NETWORK_ERROR = FetchStatus.NETWORK_ERROR

REASON_EMPTY_URL = FetchReason.EMPTY_URL
REASON_SHORTENER_FAILED = FetchReason.SHORTENER_FAILED
REASON_SHORTENER_TIMEOUT = FetchReason.SHORTENER_TIMEOUT
REASON_SHORTENER_ERROR = FetchReason.SHORTENER_ERROR
REASON_WAF_BLOCKED = FetchReason.WAF_BLOCKED
REASON_FETCH_SUCCESS = FetchReason.FETCH_SUCCESS
REASON_MEDIA_TIMEOUT = FetchReason.MEDIA_TIMEOUT
REASON_MEDIA_ERROR = FetchReason.MEDIA_ERROR
REASON_GNEWS_SNIPPET_ONLY = FetchReason.GNEWS_SNIPPET_ONLY
REASON_RSS_FULL_TEXT = FetchReason.RSS_FULL_TEXT
REASON_EXTRACT_EMPTY = FetchReason.EXTRACT_EMPTY
REASON_THREAD_CRASH = FetchReason.THREAD_CRASH
REASON_MAX_RETRIES_EXCEEDED = FetchReason.MAX_RETRIES_EXCEEDED

REASON_CATEGORY = {
    FetchReason.MEDIA_TIMEOUT: "timeout",
    FetchReason.SHORTENER_TIMEOUT: "timeout",
    FetchReason.WAF_BLOCKED: "blocked",
    FetchReason.SHORTENER_FAILED: "dead_link",
    FetchReason.EMPTY_URL: "dead_link",
    FetchReason.MEDIA_ERROR: "network_error",
    FetchReason.SHORTENER_ERROR: "network_error",
    FetchReason.THREAD_CRASH: "network_error",
    FetchReason.EXTRACT_EMPTY: "extract_empty",
    FetchReason.MAX_RETRIES_EXCEEDED: "max_retries_exceeded",
}

MIN_FULLTEXT_LENGTH = 500
FETCH_TIMEOUT = 15
PLAYWRIGHT_TIMEOUT = 15000 # ms
SLEEP_JITTER_SHORT = 2
SLEEP_JITTER_LONG = 4
MAX_RECOVERY_RETRY = 3
RECOVERY_RESOLVED = "resolved"
RECOVERY_FAILED = "failed"
RECOVERY_PENDING = "pending"
RETRYABLE_FAILURES = [
    "max_retries_exceeded", "media_timeout", "shortener_timeout", 
    "http_403", "http_429", "http_500", "http_502", "http_503",
    "media_error", "shortener_error"
]
def categorize_reason(reason: str) -> str:
    if reason in REASON_CATEGORY:
        return REASON_CATEGORY[reason]
    if str(reason).startswith("http_"):
        return "blocked" if reason in (http_reason(403), http_reason(429)) else "dead_link"
    return "other"