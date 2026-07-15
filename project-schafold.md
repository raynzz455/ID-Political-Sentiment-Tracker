ID-Political-Sentiment-Tracker/
├── apps/
│   └── pipeline/
│       ├── __init__.py
│       └── orchestrator.py       <-- (BARU) Entry point untuk jalanin semua worker
├── packages/
│   ├── shared/                   <-- (BARU) Modul bersama lintas layer
│   │   ├── __init__.py
│   │   ├── constants.py          <-- (Pindahan dari pipeline_constants.py)
│   │   ├── db_client.py          <-- (BARU) Inisialisasi Supabase client
│   │   └── logger.py             <-- (Pindahan dari pipeline_logger.py)
│   │
│   ├── ingestion/                <-- (BARU)
│   │   ├── __init__.py
│   │   ├── gnews_fetcher.py
│   │   └── ddg_fetcher.py
│   │
│   ├── enrichment/               <-- (BARU)
│   │   ├── __init__.py
│   │   ├── enricher_worker.py
│   │   └── universal_resolver.py
│   │
│   ├── validation/               <-- (BARU)
│   │   ├── __init__.py
│   │   ├── validation_worker.py
│   │   └── preprocessing_worker.py
│   │
│   ├── entity/                   <-- (BARU)
│   │   ├── __init__.py
│   │   └── entity_resolution_worker.py
│   │
│   ├── context/                  <-- (BARU)
│   │   ├── __init__.py
│   │   ├── context_worker.py
│   │   └── nlp_readiness_worker.py
│   │
│   └── nlp/                      <-- (BARU)
│       ├── __init__.py
│       ├── nlp_worker.py         <-- (Pindahan dari drain_queue.py)
│       ├── sentiment_model.py
│       └── cli_test.py
│
├── devtools/                     <-- (BARU) Untuk script testing/evaluasi
│   ├── eval/
│   │   ├── eval_metrics.py
│   │   ├── export_sentiment_ground_truth.py
│   │   └── export_relevancy_review.py
│   └── sql_tools/
│       └── check_db_stats.py
│
├── infra/
│   └── supabase/
│       └── functions/
│           └── rss-ingestion/
│               └── index.ts
├── docs/
└── .github/

                  ┌──────────────────────────────┐
                  │        RSS INGESTION         │
                  │ (RSS, Media, GNews Feed)     │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                     raw_texts (status=PENDING)
                                 │
                                 ▼
══════════════════════════════════════════════════════════════
                 LAYER 2 — ENRICHMENT
══════════════════════════════════════════════════════════════

Enricher Worker
    │
    ├── Fetch halaman asli
    ├── Trafilatura Extraction
    ├── Metadata
    └── Update

status = ENRICHED
content_type =
    FULLTEXT
    atau
    SNIPPET (GNews)

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
              LAYER 2.5 — QUALITY VALIDATION
══════════════════════════════════════════════════════════════

Validation Worker

Input:
status = ENRICHED

Melakukan:

• Quality Score
• Noise Detection
• Bahasa
• Panjang artikel
• Stopword
• Title Matching

Output:

VALIDATED
atau

FAILED

(Tidak melakukan routing lagi.)

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
             LAYER 3 — PREPROCESSING
══════════════════════════════════════════════════════════════

Preprocessing Worker

Input

status = VALIDATED

Melakukan

• Cleaning
• Unicode normalize
• Remove URL
• Remove emoji
• Remove HTML
• Sentence normalize
• dst

Output

preprocessed_text
preprocessed_at
preprocessing_version

status tetap VALIDATED

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
            LAYER 3.2 — ENTITY RESOLUTION
══════════════════════════════════════════════════════════════

Entity Resolution Worker

Input

status = VALIDATED

Menghasilkan

article_entity_map

entity_mentions

main entity

confidence

resolver source

status tetap VALIDATED

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
              LAYER 3.5 — CONTEXT
══════════════════════════════════════════════════════════════

Context Worker

Input

artikel VALIDATED
+
entity

Menghasilkan

entity_contexts

context span

window sentence

normalized context

status tetap VALIDATED

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
            LAYER 3.7 — NLP READINESS
══════════════════════════════════════════════════════════════

Readiness Worker

Mengecek apakah artikel sudah memiliki

✓ preprocessed text

✓ entity

✓ context

Jika lengkap

↓

masuk queue NLP (pgmq)

status = NLP_READY

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
             LAYER 4 — NLP PIPELINE
══════════════════════════════════════════════════════════════

Drain Queue

↓

Ambil artikel dari pgmq

↓

Fallback Sentiment

↓

Cari alias entity

↓

Relevancy Classifier

↓

Jika relevan

↓

Context Sentiment Classifier

↓

Insert sentiment_scores

↓

ACK Queue

↓

status = PROCESSED