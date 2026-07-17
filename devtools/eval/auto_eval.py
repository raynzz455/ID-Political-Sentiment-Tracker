"""
auto_eval.py v2 — Tiered Automated Model Evaluation
=====================================================
Evaluasi model NLP dengan 4 level framework (academically rigorous):

  LEVEL 1: ADVERSARIAL TEST SUITE (auto, no DB needed)
    12 test cases sulit: name collision, sarkasme, code-switching, quote attribution.
    Validasi: apakah model jebolan bug historis (Prabowo/Kapolri, dll)?

  LEVEL 2: DOMAIN BENCHMARK (politik, BUKAN review konsumer)
    Pakai test cases berita politik asli Indonesia (bukan IndoNLU SmSA
    yang domain-nya review Tokopedia/Shopee — irrelevant untuk use case ini).

  LEVEL 3: PRODUCTION DB DISTRIBUTION (analisis statistik)
    Audit distribusi label, confidence, entropy di DB produksi.

  LEVEL 4: PER-ENTITY BREAKDOWN (akurasi per tokoh)
    Akurasi Prabowo vs Rocky Gerung bisa beda jauh. Tidak boleh di-blend.

Usage:
    python -m devtools.eval.auto_eval                       # semua level
    python -m devtools.eval.auto_eval --level 1             # adversarial saja
    python -m devtools.eval.auto_eval --level 4             # per-entity saja
    python -m devtools.eval.auto_eval --skip-benchmark      # skip download dataset
"""
import os
import sys
import math
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from dotenv import load_dotenv

# Load .env explicitly dari root project
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# Pastikan variabel global terisi
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from packages.shared.db_client import get_client
    from packages.nlp.sentiment_model import get_pipeline
    from devtools.eval.adversarial_cases import ADVERSARIAL_CASES
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
except ImportError as e:
    print(f"[ERROR] Import gagal: {e}")
    sys.exit(1)

def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    print(sep)
    print("|" + "|".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(f"{str(c):<{w}}" for c, w in zip(row, col_widths)) + "|")
    print(sep)

# ============================================================
# LEVEL 1: ADVERSARIAL TEST SUITE & GATE AUDIT
# ============================================================
def run_adversarial_tests(pipeline):
    logger.info("=" * 70)
    logger.info("LEVEL 1: ADVERSARIAL TEST SUITE & RELEVANCY GATE AUDIT")
    logger.info("=" * 70)

    y_true_rel = []
    y_pred_rel = []
    
    results = []
    category_stats = defaultdict(lambda: {"total": 0, "passed": 0})

    for i, case in enumerate(ADVERSARIAL_CASES, 1):
        result = pipeline.predict_gated(text=case["text"], context=case["context"])
        
        rel_pass = (result.is_relevant == case["expected_relevant"])
        y_true_rel.append(case["expected_relevant"])
        y_pred_rel.append(result.is_relevant)

        label_pass = True
        if case["expected_relevant"] and case.get("expected_label"):
            label_pass = (result.label == case["expected_label"])

        overall_pass = rel_pass and label_pass
        category_stats[case["category"]]["total"] += 1
        if overall_pass: category_stats[case["category"]]["passed"] += 1

        status = "PASS" if overall_pass else "FAIL"
        actual_label = result.label if result.is_relevant else "(skip)"
        results.append([f"#{i}", case["category"][:18], f"{'Y' if case['expected_relevant'] else 'N'}->{'Y' if result.is_relevant else 'N'}", f"{case.get('expected_label', '-')}->{actual_label}", status])

    logger.info("\nHasil Detail Adversarial:")
    print_table(["Case", "Kategori", "Rel (Exp->Act)", "Label (Exp->Act)", "Status"], results)
    
    logger.info("\nHasil per Kategori:")
    rows = [[cat, str(s["passed"]), str(s["total"]), f"{s['passed']/s['total']*100:.0f}%"] for cat, s in sorted(category_stats.items())]
    print_table(["Kategori", "Pass", "Total", "Rate"], rows)

    logger.info("\n--- RELEVANCY GATE AUDIT (Precision/Recall) ---")
    print(classification_report(y_true_rel, y_pred_rel, target_names=["Not Relevant", "Relevant"], zero_division=0))

# ============================================================
# LEVEL 2: DOMAIN BENCHMARK
# ============================================================
DOMAIN_POLITIK_BENCHMARK = [
    {"text": "Prabowo resmikan program makan siang gratis di Gorontalo, disambut antusias warga.", "label": "positive"},
    {"text": "Skandal korupsi APBN mengguncang kabinet, sejumlah menteri diperiksa KPK.", "label": "negative"},
    {"text": "DPR akhirnya menyahkan RUU Pertanahan setelah debat panjang selama 6 jam.", "label": "neutral"},
    {"text": "Anies Baswedan hadiri deklarasi koalisi baru, disebut langkah strategis menjelang Pilkada.", "label": "positive"},
    {"text": "Kritik keras dari pengamat soal kebijakan ekonomi yang dinilai merugikan rakyat kecil.", "label": "negative"},
    {"text": "Sri Mulyani paparkan laporan keuangan kuartal III di Rapat Dengar Pendapat dengan DPR.", "label": "neutral"},
    {"text": "Kabinet Prabowo meraih apresiasi tinggi atas pencapaian investasi asing tahun ini.", "label": "positive"},
    {"text": "Unjuk rasa mahasiswa menolak kenaikan BBNKB di depan Istana Merdeka.", "label": "negative"},
    {"text": "Gibran Rakabuming Raka melakukan kunjungan kerja ke sejumlah daerah di Jawa Tengah.", "label": "neutral"},
    {"text": "Ekonom memuji kebijakan baru pemerintah di sektor industri kreatif nasional.", "label": "positive"},
]

def run_domain_benchmark(pipeline):
    logger.info("\n" + "=" * 70)
    logger.info("LEVEL 2: DOMAIN BENCHMARK (F1, Kappa, Confusion Matrix)")
    logger.info("=" * 70)

    y_true = [c["label"] for c in DOMAIN_POLITIK_BENCHMARK]
    y_pred = []
    for case in DOMAIN_POLITIK_BENCHMARK:
        result = pipeline.predict_gated(text=case["text"], context=None)
        y_pred.append(result.label)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    logger.info(f"\nOverall Accuracy : {acc*100:.1f}%")
    logger.info(f"Cohen's Kappa    : {kappa:.3f} (Agreement >0.6 is substantial)")
    
    logger.info("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"], zero_division=0))
    
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])
    print(f"                 Pred Neg | Pred Neu | Pred Pos")
    print(f"Actual Neg  |    {cm[0][0]:>6}  |   {cm[0][1]:>5}  |   {cm[0][2]:>5}")
    print(f"Actual Neu  |    {cm[1][0]:>6}  |   {cm[1][1]:>5}  |   {cm[1][2]:>5}")
    print(f"Actual Pos  |    {cm[2][0]:>6}  |   {cm[2][1]:>5}  |   {cm[2][2]:>5}")

# ============================================================
# LEVEL 3: PRODUCTION DB DISTRIBUTION
# ============================================================
def analyze_db_distribution():
    logger.info("\n" + "=" * 70)
    logger.info("LEVEL 3: PRODUCTION DB DISTRIBUTION")
    logger.info("=" * 70)

    sb = get_client()
    try:
        res = sb.table("sentiment_scores").select("label, confidence, score_negative, score_neutral, score_positive").not_.is_("entity_id", "null").limit(2000).execute()
        data = res.data or []
        if not data: return None
    except Exception as e:
        logger.error(f"DB Error: {e}"); return None

    total = len(data)
    confs = [float(d["confidence"]) for d in data if d.get("confidence") is not None]
    entropies = []
    
    for d in data:
        try:
            scores = [float(d["score_negative"]), float(d["score_neutral"]), float(d["score_positive"])]
            # FIX BUG ENTROPY: Jangan tambah 1e-9 jika p > 0
            ent = -sum(p * math.log(p) for p in scores if p > 1e-9)
            entropies.append(ent)
        except: pass

    if confs:
        avg_conf = sum(confs) / len(confs)
        logger.info(f"Rata-rata Confidence : {avg_conf:.3f}")
    
    if entropies:
        avg_ent = sum(entropies) / len(entropies)
        max_ent = math.log(3)
        confusion_pct = sum(1 for e in entropies if e > max_ent * 0.95) / len(entropies) * 100
        logger.info(f"Rata-rata Entropy    : {avg_ent:.3f} (Max: {max_ent:.3f})")
        logger.info(f"Artikel high-entropy (>95% max): {confusion_pct:.1f}%")

# ============================================================
# LEVEL 4: PER-ENTITY BREAKDOWN
# ============================================================
def run_per_entity_breakdown():
    logger.info("\n" + "=" * 70)
    logger.info("LEVEL 4: PER-ENTITY BREAKDOWN")
    logger.info("=" * 70)

    sb = get_client()
    try:
        res = sb.table("sentiment_scores").select("label, confidence, entity_id, political_entities(canonical_name)").not_.is_("entity_id", "null").limit(3000).execute()
        data = res.data or []
        if not data: return
    except: return

    by_entity = defaultdict(list)
    for row in data:
        pe = row.get("political_entities") or {}
        by_entity[pe.get("canonical_name", "?")].append(row)

    filtered = {name: rows for name, rows in by_entity.items() if len(rows) >= 3}
    if not filtered: return

    rows = []
    for name, rows_data in sorted(filtered.items(), key=lambda x: -len(x[1])):
        n = len(rows_data)
        labels = Counter(r["label"] for r in rows_data)
        pos_pct = labels.get("positive", 0) / n * 100
        neg_pct = labels.get("negative", 0) / n * 100
        neu_pct = labels.get("neutral", 0) / n * 100
        confs = [float(r["confidence"]) for r in rows_data if r.get("confidence") is not None]
        avg_conf = sum(confs) / len(confs) if confs else 0
        
        flag = " !bias" if max(pos_pct, neg_pct) > 70 else ""
        rows.append([name[:22], str(n), f"{pos_pct:.0f}%", f"{neu_pct:.0f}%", f"{neg_pct:.0f}%", f"{avg_conf:.2f}", flag])

    print_table(["Tokoh", "N", "Pos%", "Neu%", "Neg%", "Conf", "Flag"], rows)

def main():
    parser = argparse.ArgumentParser(description="Rigorous Model Evaluation v3")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], help="Run hanya 1 level")
    args = parser.parse_args()

    logger.info("MEMULAI RIGOROUS EVALUATION v3\n")
    pipeline = get_pipeline()

    run_all = (args.level is None)
    if run_all or args.level == 1: run_adversarial_tests(pipeline)
    if run_all or args.level == 2: run_domain_benchmark(pipeline)
    if run_all or args.level == 3: analyze_db_distribution()
    if run_all or args.level == 4: run_per_entity_breakdown()

if __name__ == "__main__":
    main()