### 🚀 1. Audit Tools (Post-Pipeline Quality Check)
*Tools* ini dijalankan **setiap selesai pipeline utama** untuk memantau kualitas data tanpa membebani database.

**Cek Kesehatan Database Utama:**
```bash
python -m devtools.sql_tools.check_db_status
```

**Audit Layer Enrichment (Volume & Domain):**
```bash
python -m devtools.audit.audit_enrichment
```

**Audit Layer Validation (Alasan Kegagalan):**
```bash
python -m devtools.audit.audit_validation
```

**Audit Layer Entity (Top Tokoh):**
```bash
python -m devtools.audit.audit_entity
```

---

### 🛠️ 2. Recovery Tools (Manual Heavy Recovery)
*Tools* ini dijalankan **HANYA JIKA** Audit Tools menemukan anomali (misal: tiba-tiba 1000 artikel gagal diekstrak, atau GNews mengubah enkripsinya). Ini dijalankan di komputer lokal kamu, bukan di cloud.

**A. Local GNews Playwright Resolver (Berat)**
Menggunakan Headless Chrome untuk memecahkan redirect JavaScript GNews yang tidak bisa dipecahkan oleh `requests` biasa.
```bash
python -m devtools.recovery.gnews_resolver --limit 10
```

**B. Replay Failed Articles (Mengulang Ekstraksi)**
Mengambil artikel biasa (non-GNews) yang gagal di Enrichment, mencoba men-fetch ulang, lalu mengembalikan ke `enriched` agar masuk Validation.
```bash
python -m devtools.recovery.replay_failed_articles --limit 20
```

**C. Retry Failed URLs (Network Retry)**
Mengecek apakah URL yang sebelumnya kena blokir (403/429/Timeout) sudah bisa diakses lagi. Jika bisa, dikembalikan ke `pending`.
```bash
python -m devtools.recovery.retry_failed_urls --limit 50
```

---

### 💾 3. Dataset Exporter (Untuk Training/Skripsi)
Mengekspor data bersih (Teks + Context + Entity) ke file JSON untuk keperluan Fine-Tuning model IndoBERT atau evaluasi *Ground Truth*.

```bash
python -m devtools.dataset.recover_dataset --limit 5000
```
*Ini akan menghasilkan file `dataset_ml_training.json` di root folder project.*

### 📝 Ringkasan
- **Produksi (Otomatis):** `python main.py run-prep` atau `python main.py run-nlp`
- **DevTools (Manual):** `python -m devtools.[folder].[file]`

Dengan pemisahan ini, *cloud pipeline* kamu akan 100% aman dari *crash* karena *tools* berat, sementara kamu tetap punya amunisi lengkap di lokal untuk *debugging* dan *data recovery*!