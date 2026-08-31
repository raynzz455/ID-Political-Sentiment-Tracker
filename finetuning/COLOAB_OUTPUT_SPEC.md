# Google Colab Test — Output Spec

## Yang Perlu Dijalankan di Colab

1. Upload `colab_runner.py` ke Colab
2. Set env vars (URL + SERVICE_ROLE_KEY Anda)
3. Run semua cells
4. Download 2-3 file JSON output

## Output Files yang Perlu Dikirim Balik

### 1. `colab_entity_results.json` (WAJIB)

Format:
```json
{
  "stats": {
    "v14_found": 80,
    "v14_no_entity": 20,
    "sentiment_detected": 15,
    "agree": 45,
    "disagree": 10,
    "v14_found_v12_missed": 25
  },
  "results": [
    {
      "art_id": "uuid",
      "title": "PDIP Minta Gibran Klarifikasi...",
      "v12_main": "Gibran Rakabuming Raka",
      "v14_main": "Gibran Rakabuming Raka",
      "v14_sentiment": true,
      "v14_sentiment_verbs": ["ungkap"],
      "v14_dominance": 0.167,
      "v14_in_title": true,
      "v14_count": 5,
      "v14_roles": ["nsubj", "obl"],
      "v14_all_entities": 1,
      "agree_with_v12": true
    }
  ],
  "total_time": 120.5,
  "completed": 100
}
```

**Yang saya analisa dari file ini:**
- Berapa % entity ditemukan vs v12
- Berapa % sentiment predicate terdeteksi
- Artikel mana yang v14 disagree dengan v12 (untuk verifikasi correctness)
- Distribusi roles (nsubj/obj/obl)
- Stability: apakah 100 artikel selesai tanpa crash

### 2. `colab_context_results.json` (WAJIB)

Format:
```json
{
  "results": [
    {
      "art_id": "uuid",
      "title": "Mendagri Ungkap Efisiensi APBN...",
      "entity": "Tito Karnavian",
      "v17_quality": 70,
      "v18_quality": 50,
      "v17_attr": 25,
      "v18_attr": 10,
      "root_verb": "nyata",
      "has_sentiment": false,
      "has_attribution": true,
      "is_main_actor": false,
      "ctx_text": "Menteri Dalam Negeri Muhammad Tito Karnavian menyatakan..."
    }
  ],
  "completed": 100
}
```

**Yang saya analisa dari file ini:**
- Berapa artikel downgraded (attr 25→10) — verifikasi speaker bias fix
- Berapa artikel upgraded (sentiment found, attr=40)
- Root verb distribution — apakah verb sets cukup
- Context quality comparison v17 vs v18

### 3. `colab_nlp_results.json` (OPSIONAL tapi sangat dianjurkan)

Format:
```json
[
  {
    "art_id": "uuid",
    "entity": "Prabowo Subianto",
    "label": "positive",
    "confidence": 0.892,
    "probs": [0.05, 0.05, 0.89],
    "ctx_text": "PRESIDEN Prabowo Subianto menyindir..."
  }
]
```

**Yang saya analisa dari file ini:**
- Distribusi label (positive/neutral/negative)
- Confidence distribution — untuk tuning τ threshold
- Apakah sentiment model akurat pada context v18

## Cara Mengirim

1. Download ketiga file JSON dari Colab
2. Upload/paste ke chat saya
3. Saya akan analisa dan beri verdict:
   - Apakah v14.2 entity resolution ready deploy
   - Apakah v18.1 context worker ready deploy
   - Apakah perlu tuning lebih lanjut
   - Apakah sudah layak finetuning

## Troubleshooting di Colab

**Jika Stanza crash (OOM):**
```python
# Tambahkan di akhir loop:
gc.collect()
torch.cuda.empty_cache()  # jika GPU
```

**Jika Supabase rate-limit:**
```python
# Tambah delay antar batch:
time.sleep(1)  # setiap 20 artikel
```

**Jika model download gagal:**
```python
# Pre-download model sebelum pipeline:
import stanza
stanza.download('id', processors='tokenize,pos,lemma,depparse')
```

## Yang Saya Cari di Output

### Green flags (ready deploy):
- v14_found > 70% (entity coverage bagus)
- sentiment_detected > 10% (verb sets bekerja)
- agree_with_v12 > 60% (tidak terlalu banyak breaking change)
- v18 downgraded > 20% (speaker bias fix bekerja)
- No crash in 100 articles

### Red flags (perlu fix):
- v14_found < 50% (entity coverage buruk)
- sentiment_detected < 5% (verb sets kurang)
- disagree > 30% (terlalu banyak breaking change)
- v18 downgraded = 0 (quality_score fix tidak bekerja)
- Crash before 50 articles (stability issue)
