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