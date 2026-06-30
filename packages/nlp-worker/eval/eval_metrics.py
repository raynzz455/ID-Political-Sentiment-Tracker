"""
eval_metrics.py
=================
Hitung metrik statistik PROPER (precision, recall, F1, confusion matrix)
dari ground truth yang sudah dilabeli manual. Evaluasi STAGE 1 (relevancy)
dan STAGE 2 (sentiment) TERPISAH -- jangan dicampur jadi satu angka,
karena keduanya menjawab pertanyaan berbeda.

Juga menghitung CALIBRATION CHECK: apakah confidence tinggi benar-benar
berkorelasi dengan akurasi tinggi? Ini langsung menjawab kritik soal
"ilusi high confidence" -- dengan data, bukan spekulasi.

Usage:
    python eval_metrics.py --relevancy relevancy_ground_truth_TEMPLATE.csv
    python eval_metrics.py --sentiment sentiment_ground_truth_TEMPLATE.csv
    python eval_metrics.py --relevancy <file1> --sentiment <file2>   # keduanya sekaligus

Requirement:
    pip install scikit-learn pandas --break-system-packages
"""

import sys
import argparse

try:
    import pandas as pd
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
    )
except ImportError:
    print("[ERROR] pip install scikit-learn pandas --break-system-packages")
    sys.exit(1)


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def calibration_check(df: pd.DataFrame, conf_col: str, correct_col: pd.Series, label: str):
    """
    Bucket confidence jadi beberapa bin, hitung akurasi nyata per bin.
    Kalau confidence tinggi tapi akurasi rendah -> model OVERCONFIDENT
    (persis kritik Gemini -- sekarang divalidasi dengan data, bukan spekulasi).
    """
    print(f"\n--- Calibration check: {label} ---")
    bins = [0, 0.5, 0.7, 0.85, 0.95, 1.01]
    bin_labels = ["<0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95", ">=0.95"]

    df = df.copy()
    df["_conf_bucket"] = pd.cut(df[conf_col].astype(float), bins=bins, labels=bin_labels, right=False)
    df["_correct"] = correct_col

    summary = df.groupby("_conf_bucket", observed=True).agg(
        n=("_correct", "size"),
        akurasi_aktual=("_correct", "mean"),
    )

    print(f"{'Confidence bucket':<18s} {'N':>6s} {'Akurasi aktual':>16s}")
    for bucket, row in summary.iterrows():
        flag = ""
        if bucket in (">=0.95", "0.85-0.95") and row["akurasi_aktual"] < 0.80:
            flag = "  <-- OVERCONFIDENT (confidence tinggi, akurasi rendah)"
        print(f"{bucket:<18s} {int(row['n']):>6d} {row['akurasi_aktual']*100:>14.1f}%{flag}")


def eval_relevancy(filepath: str):
    print_section(f"EVALUASI STAGE 1 — RELEVANCY GATE  ({filepath})")

    df = pd.read_csv(filepath)
    total = len(df)
    df["gold_relevant"] = df["gold_relevant"].astype(str).str.strip().str.lower()

    labeled = df[df["gold_relevant"].isin(["yes", "no"])]
    unlabeled = total - len(labeled)

    print(f"Total baris    : {total}")
    print(f"Sudah dilabeli : {len(labeled)}")
    if unlabeled > 0:
        print(f"BELUM dilabeli : {unlabeled}  <-- isi kolom gold_relevant dulu (yes/no)")

    if len(labeled) < 10:
        print("\n[STOP] Sample berlabel terlalu sedikit (<10) untuk metrik bermakna.")
        return

    gold = labeled["gold_relevant"].map({"yes": "relevant", "no": "not_relevant"})
    pred = labeled["gate_decision"]

    print(f"\nDistribusi gold label : {gold.value_counts().to_dict()}")
    print(f"Distribusi prediksi   : {pred.value_counts().to_dict()}")

    print("\n--- Classification Report ---")
    print(classification_report(gold, pred, zero_division=0))

    print("--- Confusion Matrix ---")
    labels = ["relevant", "not_relevant"]
    cm = confusion_matrix(gold, pred, labels=labels)
    print(f"{'':>16s} {'pred:relevant':>15s} {'pred:not_relevant':>18s}")
    for i, lbl in enumerate(labels):
        print(f"actual:{lbl:<10s}{cm[i][0]:>15d} {cm[i][1]:>18d}")

    correct = (gold == pred)
    calibration_check(labeled, "relevancy_confidence", correct, "Relevancy Gate")

    n_not_relevant_gold = (gold == "not_relevant").sum()
    if n_not_relevant_gold == 0:
        print("\n[PERINGATAN] Tidak ada satupun ground truth 'not_relevant' di sample ini.")
        print("             Recall untuk kelas not_relevant TIDAK BISA dihitung valid.")
        print("             Tambahkan lebih banyak kasus false-positive yang dicurigai")
        print("             (seperti kasus Kapolri vs Presiden Prabowo) ke sample berikutnya.")


def eval_sentiment(filepath: str):
    print_section(f"EVALUASI STAGE 2 — SENTIMENT CLASSIFIER  ({filepath})")

    df = pd.read_csv(filepath)
    total = len(df)
    df["gold_label"] = df["gold_label"].astype(str).str.strip().str.lower()

    valid_labels = {"negative", "neutral", "positive"}
    labeled = df[df["gold_label"].isin(valid_labels)]
    unlabeled = total - len(labeled)

    print(f"Total baris    : {total}")
    print(f"Sudah dilabeli : {len(labeled)}")
    if unlabeled > 0:
        print(f"BELUM dilabeli : {unlabeled}  <-- isi kolom gold_label dulu (negative/neutral/positive)")

    if len(labeled) < 10:
        print("\n[STOP] Sample berlabel terlalu sedikit (<10) untuk metrik bermakna.")
        return

    gold = labeled["gold_label"]
    pred = labeled["predicted_label"]

    print(f"\nDistribusi gold label : {gold.value_counts().to_dict()}")
    print(f"Distribusi prediksi   : {pred.value_counts().to_dict()}")
    print(f"\nAkurasi keseluruhan: {accuracy_score(gold, pred)*100:.1f}%")

    print("\n--- Classification Report (per kelas) ---")
    print(classification_report(gold, pred, zero_division=0))

    print("--- Confusion Matrix ---")
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(gold, pred, labels=labels)
    header = "".join(f"{f'pred:{l}':>15s}" for l in labels)
    print(f"{'':>16s}{header}")
    for i, lbl in enumerate(labels):
        row = "".join(f"{cm[i][j]:>15d}" for j in range(len(labels)))
        print(f"actual:{lbl:<10s}{row}")

    correct = (gold == pred)
    calibration_check(labeled, "predicted_confidence", correct, "Sentiment Classifier")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi statistik pipeline 2-stage")
    parser.add_argument("--relevancy", type=str, help="Path ke relevancy_ground_truth CSV (sudah dilabeli)")
    parser.add_argument("--sentiment", type=str, help="Path ke sentiment_ground_truth CSV (sudah dilabeli)")
    args = parser.parse_args()

    if not args.relevancy and not args.sentiment:
        print("[ERROR] Berikan minimal salah satu: --relevancy <file> atau --sentiment <file>")
        sys.exit(1)

    if args.relevancy:
        eval_relevancy(args.relevancy)

    if args.sentiment:
        eval_sentiment(args.sentiment)

    print(f"\n{'=' * 70}")
    print("SELESAI.")
    print("=" * 70)


if __name__ == "__main__":
    main()
