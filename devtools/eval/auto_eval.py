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

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from packages.shared.db_client import get_client
    from packages.nlp.sentiment_model import get_pipeline
    from devtools.eval.adversarial_cases import (
        ADVERSARIAL_CASES,
        get_cases_by_category,
        get_categories,
    )
except ImportError as e:
    print(f"[ERROR] Import gagal: {e}")
    print("Pastikan struktur packages/ dan devtools/ ada.")
    sys.exit(1)


# ============================================================
# HELPER: format tabel rapi
# ============================================================
def print_table(headers, rows, col_widths=None):
    """Print tabel sederhana."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_line = "|" + "|".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + "|"
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print("|" + "|".join(f"{str(c):<{w}}" for c, w in zip(row, col_widths)) + "|")
    print(sep)


# ============================================================
# LEVEL 1: ADVERSARIAL TEST SUITE
# ============================================================
def run_adversarial_tests(pipeline):
    """Jalankan 12 test cases sulit, hitung pass rate per kategori."""
    logger.info("=" * 70)
    logger.info("LEVEL 1: ADVERSARIAL TEST SUITE (12 cases, 6 kategori)")
    logger.info("=" * 70)

    results = []
    category_stats = defaultdict(lambda: {"total": 0, "passed": 0})

    for i, case in enumerate(ADVERSARIAL_CASES, 1):
        text = case["text"]
        context = case["context"]
        expected_rel = case["expected_relevant"]
        expected_label = case.get("expected_label")
        cat = case["category"]

        result = pipeline.predict_gated(text=text, context=context)

        rel_pass = (result.is_relevant == expected_rel)

        label_pass = True
        if expected_rel and expected_label:
            label_pass = (result.label == expected_label)

        overall_pass = rel_pass and label_pass
        category_stats[cat]["total"] += 1
        if overall_pass:
            category_stats[cat]["passed"] += 1

        status = "PASS" if overall_pass else "FAIL"
        actual_label = result.label if result.is_relevant else "(skip)"
        results.append([
            f"#{i}",
            cat[:18],
            (context or "(none)")[:18],
            f"{'Y' if expected_rel else 'N'}->{'Y' if result.is_relevant else 'N'}",
            f"{expected_label or '-'}->{actual_label}",
            status,
        ])

        if not overall_pass:
            logger.info(f"  [{status}] {case['note'][:60]}")
            logger.info(f"           conf={result.relevancy_confidence:.2f} "
                        f"label={actual_label} | expected: rel={expected_rel}, lbl={expected_label}")

    logger.info("\nHasil per kategori:")
    rows = [[cat, str(s["passed"]), str(s["total"]),
             f"{s['passed']/s['total']*100:.0f}%"]
            for cat, s in sorted(category_stats.items())]
    print_table(["Kategori", "Pass", "Total", "Rate"], rows)

    total_pass = sum(s["passed"] for s in category_stats.values())
    total_all = sum(s["total"] for s in category_stats.values())
    logger.info(f"\n>>> ADVERSARIAL OVERALL: {total_pass}/{total_all} "
                f"({total_pass/total_all*100:.0f}%) <<<")

    if total_pass / total_all < 0.8:
        logger.warning("Pass rate < 80%. Model masih lemah di edge cases.")
        logger.warning("Pertimbangkan: tuning RELEVANCY_THRESHOLD atau tambah training data.")


# ============================================================
# LEVEL 2: DOMAIN BENCHMARK (politik Indonesia)
# ============================================================
# Ground truth mini: 10 artikel politik Indonesia berlabel manual.
# INI BUKAN IndoNLU SmSA (review konsumer) — ini domain politik asli.
DOMAIN_POLITIK_BENCHMARK = [
    {"text": "Prabowo resmikan program makan siang gratis di Gorontalo, disambut antusias warga.",
     "label": "positive"},
    {"text": "Skandal korupsi APBN mengguncang kabinet, sejumlah menteri diperiksa KPK.",
     "label": "negative"},
    {"text": "DPR akhirnya menyahkan RUU Pertanahan setelah debat panjang selama 6 jam.",
     "label": "neutral"},
    {"text": "Anies Baswedan hadiri deklarasi koalisi baru, disebut langkah strategis menjelang Pilkada.",
     "label": "positive"},
    {"text": "Kritik keras dari pengamat soal kebijakan ekonomi yang dinilai merugikan rakyat kecil.",
     "label": "negative"},
    {"text": "Sri Mulyani paparkan laporan keuangan kuartal III di Rapat Dengar Pendapat dengan DPR.",
     "label": "neutral"},
    {"text": "Kabinet Prabowo meraih apresiasi tinggi atas pencapaian investasi asing tahun ini.",
     "label": "positive"},
    {"text": "Unjuk rasa mahasiswa menolak kenaikan BBNKB di depan Istana Merdeka.",
     "label": "negative"},
    {"text": "Gibran Rakabuming Raka melakukan kunjungan kerja ke sejumlah daerah di Jawa Tengah.",
     "label": "neutral"},
    {"text": "Ekonom memuji kebijakan baru pemerintah di sektor industri kreatif nasional.",
     "label": "positive"},
]


def run_domain_benchmark(pipeline):
    """Test akurasi model di domain politik Indonesia."""
    logger.info("\n" + "=" * 70)
    logger.info("LEVEL 2: DOMAIN BENCHMARK — Berita Politik Indonesia (10 cases)")
    logger.info("=" * 70)
    logger.info("(Mini benchmark. Untuk skripsi: perlu 200+ labeled data politik.)\n")

    correct = 0
    results = []
    for case in DOMAIN_POLITIK_BENCHMARK:
        result = pipeline.predict_gated(text=case["text"], context=None)
        is_correct = (result.label == case["label"])
        if is_correct:
            correct += 1
        results.append([
            case["label"],
            result.label,
            f"{result.sentiment_confidence:.2f}",
            "OK" if is_correct else "X",
        ])

    print_table(["Expected", "Predicted", "Conf", "OK"], results)

    acc = correct / len(DOMAIN_POLITIK_BENCHMARK) * 100
    logger.info(f"\n>>> DOMAIN ACCURACY: {acc:.1f}% "
                f"({correct}/{len(DOMAIN_POLITIK_BENCHMARK)}) <<<")

    if acc < 70:
        logger.warning("Akurasi domain < 70%. Cek apakah model tepat untuk berita politik.")
    elif acc >= 85:
        logger.info("Akurasi domain >= 85%. Model cukup akurat di domain ini.")


# ============================================================
# LEVEL 3: PRODUCTION DB DISTRIBUTION
# ============================================================
def analyze_db_distribution():
    """Analisis distribusi statistik di DB produksi."""
    logger.info("\n" + "=" * 70)
    logger.info("LEVEL 3: PRODUCTION DB DISTRIBUTION (analisis statistik)")
    logger.info("=" * 70)

    sb = get_client()
    try:
        res = sb.table("sentiment_scores") \
                .select("label, confidence, score_negative, score_neutral, score_positive, entity_id") \
                .limit(2000) \
                .execute()
        data = res.data or []
        if not data:
            logger.warning("Tidak ada data di sentiment_scores. Jalankan NLP Worker dulu.")
            return None
    except Exception as e:
        logger.error(f"Gagal query DB: {e}")
        return None

    entity_data = [d for d in data if d.get("entity_id")]

    if not entity_data:
        logger.warning("Tidak ada data entity-level (hanya fallback NULL). "
                       "Jalankan entity resolution worker dulu.")
        return None

    total = len(entity_data)
    logger.info(f"Total sample entity-level: {total}")

    label_counts = Counter(d["label"] for d in entity_data)
    logger.info("\nDistribusi label (entity-level):")
    for label in ["positive", "neutral", "negative"]:
        c = label_counts.get(label, 0)
        pct = c / total * 100
        bar = "#" * int(pct / 2.5)
        logger.info(f"  {label:10s} {c:4d} ({pct:5.1f}%) {bar}")

    confs = [float(d["confidence"]) for d in entity_data if d.get("confidence") is not None]
    if confs:
        avg_conf = sum(confs) / len(confs)
        sorted_confs = sorted(confs)
        median = sorted_confs[len(sorted_confs) // 2]
        logger.info(f"\nConfidence: avg={avg_conf:.3f}, median={median:.3f}")
        if avg_conf > 0.80:
            logger.info("  [Status] Cukup tinggi. TAPI cek Level 4 — bisa overconfident.")
        elif avg_conf < 0.60:
            logger.warning("  [Status] Rendah. Model banyak ragu.")
        else:
            logger.info("  [Status] Sedang. Reasonable.")

    entropies = []
    for d in entity_data:
        try:
            scores = [float(d["score_negative"]), float(d["score_neutral"]), float(d["score_positive"])]
            ent = -sum(p * math.log(p + 1e-9) for p in scores if p > 0)
            entropies.append(ent)
        except (KeyError, TypeError, ValueError):
            pass
    if entropies:
        avg_ent = sum(entropies) / len(entropies)
        max_ent = math.log(3)
        confusion_pct = sum(1 for e in entropies if e > max_ent * 0.95) / len(entropies) * 100
        logger.info(f"\nEntropy: avg={avg_ent:.3f} (max={max_ent:.3f})")
        logger.info(f"Artikel dengan entropy kritis (>95% max): {confusion_pct:.1f}%")
        if confusion_pct > 25:
            logger.warning(">25% artikel confusion tinggi. Distribusi bisa tidak reliable.")

    return entity_data


# ============================================================
# LEVEL 4: PER-ENTITY BREAKDOWN
# ============================================================
def run_per_entity_breakdown():
    """Akurasi/distribusi per tokoh — tidak di-blend."""
    logger.info("\n" + "=" * 70)
    logger.info("LEVEL 4: PER-ENTITY BREAKDOWN (distribusi per tokoh)")
    logger.info("=" * 70)

    sb = get_client()
    try:
        res = sb.table("sentiment_scores") \
                .select("label, confidence, entity_id, political_entities(canonical_name)") \
                .not_("entity_id", "is", None) \
                .limit(3000) \
                .execute()
        data = res.data or []
        if not data:
            logger.warning("Tidak ada data entity-level di DB.")
            return
    except Exception as e:
        logger.error(f"Gagal query: {e}")
        return

    by_entity = defaultdict(list)
    for row in data:
        pe = row.get("political_entities") or {}
        name = pe.get("canonical_name", "?")
        by_entity[name].append(row)

    MIN_SAMPLES = 3
    filtered = {name: rows for name, rows in by_entity.items() if len(rows) >= MIN_SAMPLES}

    if not filtered:
        logger.warning(f"Tidak ada tokoh dengan >= {MIN_SAMPLES} sentiment scores.")
        logger.warning("Jalankan NLP worker lebih lama untuk kumpulkan data per-tokoh.")
        return

    logger.info(f"\nTokoh dengan >= {MIN_SAMPLES} scores: {len(filtered)}")
    logger.info(f"(Skip tokoh dengan < {MIN_SAMPLES} — terlalu sedikit untuk statistik)\n")

    rows = []
    for name, rows_data in sorted(filtered.items(), key=lambda x: -len(x[1])):
        n = len(rows_data)
        labels = Counter(r["label"] for r in rows_data)
        pos_pct = labels.get("positive", 0) / n * 100
        neg_pct = labels.get("negative", 0) / n * 100
        neu_pct = labels.get("neutral", 0) / n * 100
        confs = [float(r["confidence"]) for r in rows_data if r.get("confidence") is not None]
        avg_conf = sum(confs) / len(confs) if confs else 0

        flag = ""
        if max(pos_pct, neg_pct) > 70:
            flag = " !bias"

        rows.append([
            name[:22],
            str(n),
            f"{pos_pct:.0f}%",
            f"{neu_pct:.0f}%",
            f"{neg_pct:.0f}%",
            f"{avg_conf:.2f}",
            flag,
        ])

    print_table(
        ["Tokoh", "N", "Pos%", "Neu%", "Neg%", "Conf", "Flag"],
        rows,
    )
    logger.info("\nKeterangan:")
    logger.info("  !bias = salah satu label >70% (distribusi skewed, mungkin model bias)")
    logger.info("  Conf = rata-rata confidence score (0-1)")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Tiered Model Evaluation v2")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4],
                        help="Run hanya 1 level (default: semua)")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip Level 2 (domain benchmark)")
    args = parser.parse_args()

    logger.info("MEMULAI TIERED MODEL EVALUATION v2\n")

    logger.info("Loading pipeline (3 model IndoBERT)...")
    pipeline = get_pipeline()
    logger.info("Pipeline ready.\n")

    run_all = (args.level is None)

    if run_all or args.level == 1:
        run_adversarial_tests(pipeline)

    if (run_all or args.level == 2) and not args.skip_benchmark:
        run_domain_benchmark(pipeline)

    if run_all or args.level == 3:
        analyze_db_distribution()

    if run_all or args.level == 4:
        run_per_entity_breakdown()

    logger.info("\n" + "=" * 70)
    logger.info("EVALUASI SELESAI.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
