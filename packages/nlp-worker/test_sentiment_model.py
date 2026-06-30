"""
test_sentiment_model.py — Verifikasi 2-stage pipeline (relevancy + sentiment)
================================================================================
v2: sekarang test relevancy gate secara eksplisit -- ini metrik yang benar
untuk kasus false-positive entity matching (Kapolri vs Presiden Prabowo),
BUKAN sentiment confidence seperti di v1 (yang salah desain).

Usage:
    pip install torch transformers --break-system-packages
    python test_sentiment_model.py
"""

from sentiment_model import get_pipeline


TEST_CASES = [
    {
        "text": "Prabowo resmikan program makan siang gratis di Gorontalo, "
                "disambut antusias warga setempat",
        "context": "Prabowo Subianto",
        "expected_relevant": True,
        "note": "Benar-benar tentang Presiden Prabowo -> harus RELEVAN",
    },
    {
        "text": "PDIP minta Gibran klarifikasi soal dugaan aliran dana ke "
                "demo mahasiswa BEM UBK",
        "context": "Gibran Rakabuming Raka",
        "expected_relevant": True,
        "note": "Benar-benar tentang Gibran -> harus RELEVAN",
    },
    {
        "text": "Kapolri Jenderal Listyo Sigit Prabowo menyerahkan 6.000 "
                "bantuan sosial dalam rangka HUT Bhayangkara ke-80",
        "context": "Prabowo Subianto",
        "expected_relevant": False,
        "note": (
            "TEST UTAMA: ini tentang Kapolri (orang BERBEDA), bukan Presiden. "
            "Relevancy gate HARUS bilang TIDAK relevan -- ini test yang benar "
            "untuk kasus false-positive regex matching, bukan sentiment confidence."
        ),
    },
    {
        "text": "DPRD Kota Bandung sahkan Raperda Pencegahan Perilaku Seksual Berisiko",
        "context": None,
        "expected_relevant": True,  # N/A, pakai fallback
        "note": "Tidak ada entity context -> fallback document-level (tidak di-gate)",
    },
    {
        "text": "Anies Baswedan hadiri deklarasi koalisi baru, disebut sejumlah "
                "pengamat sebagai langkah strategis menjelang Pilkada",
        "context": "Anies Baswedan",
        "expected_relevant": True,
        "note": "Benar-benar tentang Anies -> harus RELEVAN",
    },
    {
        "text": "Identitas mayat perempuan dalam mobil pelat merah di parkiran "
                "Bandara Juanda berhasil diungkap polisi",
        "context": "Ridwan Kamil",
        "expected_relevant": False,
        "note": (
            "TEST TAMBAHAN: kasus false-positive lama dari regex (alias 'RK'/'Emil' "
            "match artikel kriminal tidak terkait). Relevancy gate harus bilang "
            "TIDAK relevan."
        ),
    },
]


def main():
    print("=" * 78)
    print("VERIFIKASI 2-STAGE PIPELINE: RELEVANCY GATE + SENTIMENT")
    print("=" * 78)
    print("Stage 1: apriandito/indobert-relevancy-classifier")
    print("Stage 2: apriandito/indobert-sentiment-classifier (hanya jika relevan)")
    print("Fallback: taufiqdp/indonesian-sentiment (jika context=None)")
    print("=" * 78)

    pipeline = get_pipeline()
    correct = 0
    total_gated = 0  # exclude case context=None dari skor akurasi gate

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}]")
        print(f"  Text    : {case['text'][:85]}{'...' if len(case['text']) > 85 else ''}")
        print(f"  Context : {case['context'] or '(tidak ada -> fallback)'}")

        result = pipeline.predict_gated(case["text"], case["context"])

        if case["context"] is not None:
            total_gated += 1
            match = result.is_relevant == case["expected_relevant"]
            correct += int(match)
            status = "✓ SESUAI" if match else "✗ TIDAK SESUAI EKSPEKTASI"

            print(f"  → Relevan?        : {result.is_relevant} (confidence={result.relevancy_confidence:.3f})")
            print(f"  → Ekspektasi      : {case['expected_relevant']}  [{status}]")

        if result.is_relevant and result.label:
            print(f"  → Sentiment       : {result.label} (conf={result.sentiment_confidence:.3f})")
            print(f"  → Scores          : neg={result.scores[0]:.3f} neu={result.scores[1]:.3f} pos={result.scores[2]:.3f}")
        else:
            print(f"  → Sentiment       : (di-skip, tidak relevan -- ini PERILAKU YANG BENAR)")

        print(f"  Catatan           : {case['note']}")

    print(f"\n{'=' * 78}")
    print(f"HASIL RELEVANCY GATE: {correct}/{total_gated} sesuai ekspektasi")
    print("=" * 78)
    if correct == total_gated:
        print("Semua kasus relevancy SESUAI ekspektasi. Pipeline siap dipatch ke cli_test.py")
    else:
        print("Ada ketidaksesuaian. JANGAN lanjut wire ke production dulu --")
        print("periksa RELEVANCY_THRESHOLD di sentiment_model.py atau cek")
        print("id2label model (kemungkinan label terbalik, lihat log loading di atas).")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
