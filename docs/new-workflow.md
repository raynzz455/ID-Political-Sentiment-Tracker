# SensiBanget-ID Pipeline (Ideal)

```text
                        ========================
                        DATA SOURCES
                        ========================

RSS
DDG
Manual
Historical Backfill

            │
            ▼

──────────────────────────────────────────────
LAYER 1
INGESTION
──────────────────────────────────────────────

ingestion_worker

↓

raw_texts
status=pending

↓

──────────────────────────────────────────────
LAYER 2
ENRICHMENT
──────────────────────────────────────────────

enricher_worker

↓

Universal Resolver

↓

Trafilatura

↓

raw_texts

status=enriched

↓

──────────────────────────────────────────────
LAYER 2.6
VALIDATION
──────────────────────────────────────────────

validation_worker

↓

score >= threshold

↓

raw_texts

status=validated

↓

──────────────────────────────────────────────
LAYER 3
ENTITY RESOLUTION
──────────────────────────────────────────────

entity_resolution_worker

↓

SpaCy NER

↓

entity_resolver

↓

article_entity_map

↓

entity_mentions

↓

entity_resolved_at = NOW()

(status tetap validated)

↓

──────────────────────────────────────────────
LAYER 4
NLP
──────────────────────────────────────────────

nlp_worker

↓

Loop setiap entity

↓

Context Extraction

↓

IndoBERT Relevancy

↓

IndoBERT Sentiment

↓

sentiment_scores

↓

raw_texts.status = processed

↓

──────────────────────────────────────────────
LAYER 5
AUTO DISCOVERY
──────────────────────────────────────────────

auto_discovery_worker

↓

Cari entity yang:

- belum lengkap
- entity baru
- confidence rendah
- metadata kosong

↓

Discovery Queue

↓

Web Search

Wikipedia

Wikidata

KPU

DPR

Partai

News

↓

Candidate Profile

↓

Review Rule

↓

political_entities

↓

entity_aliases

↓

discovery_history
```

---

# Historical Backfill

```text
political_entities

↓

pilih entity

↓

historical_backfill_worker

↓

Cari artikel lama

↓

raw_texts

status=pending

↓

Pipeline berjalan NORMAL

Enrichment

↓

Validation

↓

Entity Resolution

↓

NLP

↓

processed
```

---

# Hubungan antar Worker

```text
                ingestion_worker
                        │
                        ▼
               enricher_worker
                        │
                        ▼
              validation_worker
                        │
                        ▼
         entity_resolution_worker
                        │
                        ▼
                  nlp_worker
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
historical_backfill_worker    auto_discovery_worker
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
                  political_entities
```

## Tugas masing-masing

| Worker | Fungsi |
|---------|---------|
| ingestion_worker | Mengambil URL dari RSS/DDG/API/Manual |
| enricher_worker | Resolve URL dan mengambil isi artikel |
| validation_worker | Menilai kualitas artikel |
| entity_resolution_worker | Menemukan semua tokoh dalam artikel |
| entity_resolver | Menyatukan alias menjadi entity yang sama |
| nlp_worker | Menghitung relevansi dan sentimen tiap entity |
| historical_backfill_worker | Mengambil artikel lama berdasarkan entity |
| auto_discovery_worker | Melengkapi data entity (foto, Wikipedia, partai, jabatan, alias, dsb.) |