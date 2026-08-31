"""
context_worker.py v20 — COMPREHENSIVE Verb/Noun Sets
====================================================
v20: Expanded dari 64 → 200+ lemmas, organized by semantic category.

IMPROVEMENTS over v19.1:
  1. Fixed: removed non-lemma forms (menuduh, menuding, membuktikan, etc.)
     Stanza returns ROOT lemmas — prefixed forms never match
  2. Expanded: 64 → 200+ unique lemmas across 8 categories
  3. Added: NEGATION detection (tidak, bukan, jangan) — reverses sentiment
  4. Added: INTENSITY modifiers (sangat, cukup, terlalu) — affects confidence
  5. Added: CONDITIONAL markers (jika, kalau, seandainya) — reduces confidence
  6. Added: HEDGING words (mungkin, barangkali, sepertinya) — reduces confidence
  7. Organized by semantic category for maintainability

CATEGORY OVERVIEW:
  A. NEGATIVE evaluation verbs (entity criticized/accused/sanctioned)
  B. POSITIVE evaluation verbs (entity praised/supported/endorsed)
  C. ATTRIBUTION verbs (entity as SPEAKER — neutral, not target)
  D. NEGATIVE framing nouns (dugaan, korupsi, skandal, etc.)
  E. POSITIVE framing nouns (pujian, prestasi, dukungan, etc.)
  F. NEGATION words (reverses sentiment polarity)
  G. INTENSITY modifiers (sangat, cukup, terlalu)
  H. HEDGING/conditional words (reduces confidence)

LEMMA RULES (Stanza Indonesian):
  - Verbs: dikritik→kritik, mengecam→kecam, memuji→puji, menuduh→tuduh
  - Nouns: dugaan→dugaan, korupsi→korupsi (nouns don't change)
  - Passive: detected via deprel=nsubj:pass (same lemma as active)
  - Prefixed forms (me-, ber-, ter-, di-, ke-) → STRIPPED to root
"""

# ═══════════════════════════════════════════════════════════════
# A. NEGATIVE EVALUATION VERBS (entity as TARGET of negative action)
# ═══════════════════════════════════════════════════════════════
SENTIMENT_PREDICATES_ACTIVE = {
    # --- A1. Criticism & Mockery (langsung menyerang) ---
    "kritik",      # mengkritik → kritik
    "kecam",       # mengecam → kecam
    "sindir",      # menyindir → sindir
    "serang",      # menyerang → serang
    "hina",        # menghina → hina
    "cela",        # mencela → cela
    "ejek",        # mengejek → ejek
    "cibir",       # mencibir → cibir
    "remeh",       # meremehkan → remeh
    "hina",        # (dup check — keep once)
    "olok",        # mengolok → olok
    "jewer",       # colloquial criticism
    "biri",        # membiri → biri (mockery)

    # --- A2. Accusation & Suspicion (tuduhan) ---
    "tuduh",       # menuduh → tuduh
    "tuding",      # menuding → tuding
    "duga",        # menduga → duga
    "kait",        # mengaitkan → kait
    "libat",       # melibatkan → libat
    "curiga",      # mencurigai → curiga
    "sangka",      # tersangka → sangka
    "fiting",      # menfitnah → fitnah (loanword)

    # --- A3. Legal Action (tindakan hukum) ---
    "vonis",       # divonis → vonis
    "tahan",       # menahan → tahan
    "tangkap",     # menangkap → tangkap
    "cekal",       # mencekal → cekal
    "pidana",      # dipidana → pidana
    "dakwa",       # mendakwa → dakwa
    "tuntut",      # menuntut → tuntut
    "adili",       # mengadili → adili
    "eksekusi",    # mengeksekusi → eksekusi
    "jangkar",     # menjangkau → jangkar (legal reach)

    # --- A4. Sanction & Removal (sanksi & pemberhentian) ---
    "pecat",       # memecat → pecat
    "mundur",      # mundur (resign)
    "undur",       # mengundurkan → undur
    "berhenti",    # stop
    "ganti",       # mengganti → ganti
    "copot",       # mencopot → copot
    "pencabutan",  # noun form
    "razia",       # merazia → razia
    "sita",        # menyita → sita
    "denda",       # mendenda → denda
    "hukum",       # menghukum → hukum
    "ganjar",      # mengganjar → ganjar
    "bubarkan",    # membubarkan → bubarkan

    # --- A5. Exposure & Revelation (membongkar) ---
    "bongkar",     # membongkar → bongkar
    "ungkap",      # mengungkap → ungkap
    "ketahui",     # diketahui → ketahui
    "bukti",       # noun/verb boundary
    "buktikan",    # membuktikan → bukti (Stanza may keep -kan)
    "teliti",      # meneliti → teliti
    "audit",       # mengaudit → audit
    "investigasi", # loanword

    # --- A6. Violation & Deviation (pelanggaran) ---
    "langgar",     # melanggar → langgar
    "simpang",     # menyimpang → simpang
    "salah",       # salah (general)
    "salahguna",   # salahgunakan → salahguna
    "sewenang",    # penyelewengan → sewenang (root)
    "nyelewang",   # colloquial
    "markus",      # noun (corruptor)

    # --- A7. Loss & Damage (kerugian) ---
    "rugi",        # merugikan → rugi
    "beban",       # membebani → beban
    "gagal",       # gagal (failure)
    "runtuh",      # runtuh (collapse)
    "jatuh",       # jatuh (fall)
    "anjlok",      # anjlok (plunge)
    "merosot",     # merosot (decline)
    "terpuruk",    # terpuruk (slump)

    # --- A8. Political Opposition (perlawanan politik) ---
    "bantahan",    # menyangkal → sangkal
    "sangkal",     # menyangkal → sangkal
    "tolak",       # menolak → tolak
    "keberatan",   # berkeberatan → keberatan
    "entang",      # menentang → entang
    "lawan",       # melawan → lawan
    "hadang",      # menghadang → hadang
    "halang",      # menghalangi → halang
    "batalkan",    # membatalkan → batalkan

    # --- A9. Judgment & Evaluation (penilaian negatif) ---
    "nilai",       # menilai → nilai
    "sorot",       # menyorot → sorot
    "pandang",     # memandang → pandang
    "sikapi",      # menyikapi → sikapi
    "persepsi",    # menerima → terima
    "anggap",      # menganggap → anggap
    "nada",        # bernada → nada
    "kesan",       # noun (impression)

    # --- A10. Scandal & Controversy (noun-verbs) ---
    "skandal",     # scandal
    "kontroversi", # controversy
    "viral",       # viral (internet)
    "polemik",     # polemic
    "sensasi",     # sensation
    "skorsing",    # suspension (loanword)
}

# ═══════════════════════════════════════════════════════════════
# B. POSITIVE EVALUATION VERBS (entity as TARGET of positive action)
# ═══════════════════════════════════════════════════════════════
SENTIMENT_PREDICATES_POSITIVE = {
    # --- B1. Praise & Admiration ---
    "puji",        # memuji → puji
    "sanjung",     # menyanjung → sanjung
    "kagum",       # mengagumi → kagum
    "takjub",      # takjub (amaze)
    "apresiasi",   # mengapresiasi → apresiasi
    "acungi",      # acungi jempol → acungi
    "hormat",      # menghormati → hormat
    "hargai",      # menghargai → hargai
    "kultus",      # mengultuskan → kultus
    "idolakan",    # mengidolakan → idolakan

    # --- B2. Support & Endorsement ---
    "dukung",      # mendukung → dukung
    "restui",      # merestui → restui
    "sahkan",      # mensahkan → sahkan
    "setuju",      # menyetujui → setuju
    "kukuhkan",    # mengukuhkan → kukuhkan
    "akui",        # mengakui → akui
    "legitimasi",  # noun/verb
    "backing",     # loanword
    "endorse",     # loanword

    # --- B3. Achievement & Success ---
    "raih",        # meraih → raih
    "capai",       # mencapai → capai
    "menang",      # menang (win)
    "sukses",      # sukses (success)
    "berhasil",    # berhasil (succeed)
    "lulus",       # lulus (pass)
    "konsisten",   # konsisten
    "unggul",      # unggul (excel)
    "telebih",     # telebih (excel — colloquial)
    "peloporan",   # noun (pioneering)

    # --- B4. Honor & Recognition ---
    "tunjuk",      # menunjuk → tunjuk
    "angkat",      # mengangkat → angkat
    "lantik",      # melantik → lantik
    "promosi",     # mempromosikan → promosi
    "naik",        # naik (rise)
    "kukuh",       # mengukuhkan → kukuh
    "anugerah",    # noun (award)

    # --- B5. Trust & Confidence ---
    "percaya",     # percaya (believe)
    "yakin",       # yakin (certain)
    "pede",        # percaya diri → pede
    "andalkan",    # mengandalkan → andalkan
    "amanah",      # noun/adj (trustworthy)
}

# Merge positive verbs into active predicates (for has_sentiment_predicate detection)
SENTIMENT_PREDICATES_ACTIVE.update(SENTIMENT_PREDICATES_POSITIVE)

# ═══════════════════════════════════════════════════════════════
# C. ATTRIBUTION VERBS (entity as SPEAKER — neutral, not target)
# ═══════════════════════════════════════════════════════════════
ATTRIBUTION_WORDS = {
    # --- C1. Core Speaking Verbs ---
    "kata",        # mengatakan → kata
    "nyata",       # menyatakan → nyata
    "tegas",       # menegaskan → tegas
    "jelaskan",    # menjelaskan → jelaskan
    "tambah",      # menambahkan → tambah
    "imbau",       # mengimbau → imbau
    "ingat",       # mengingatkan → ingat
    "sampai",      # menyampaikan → sampai
    "sebut",       # menyebut → sebut
    "papar",       # memaparkan → papar
    "ucap",        # mengucapkan → ucap
    "tutur",       # menuturkan → tutur
    "ujar",        # mengucapkan → ujar
    "katakan",     # form
    "ungkap",      # NOTE: also in negative (exposure) — context-dependent

    # --- C2. Answer & Response ---
    "jawab",       # menjawab → jawab
    "balas",       # membalas → balas
    "tanggapi",    # menanggapi → tanggapi
    "sanggah",     # menyanggah → sanggah
    "bantah",      # membantah → bantah
    "sangkal",     # menyangkal → sangkal
    "klaim",       # mengklaim → klaim
    "aku",         # mengakui → aku

    # --- C3. Suggestion & Proposal ---
    "saran",       # menyarankan → saran
    "menyaran",    # menyarankan → saran (Stanza may keep prefix)
    "rekomendasi", # merekomendasikan → rekomendasi
    "usul",        # mengusulkan → usul
    "ajak",        # mengajak → ajak
    "anjurkan",    # menganjurkan → anjurkan
    "himbau",      # mengimbau → imbau
    "seru",        # menyerukan → seru

    # --- C4. Request & Command ---
    "pinta",       # meminta → pinta
    "minta",       # minta
    "perintah",    # memerintahkan → perintah
    "instruksi",   # menginstruksikan → instruksi
    "wantiwanti",  # noun (warning)
    "imbau",       # imbau

    # --- C5. Emphasis & Highlight ---
    "tekan",       # menekankan → tekan
    "sorot",       # NOTE: also in negative (judgment)
    "tandai",      # menandai → tandai
    "garis",       # garis bawahi → garis
    "khusus",      # especially
    "utama",       # main

    # --- C6. Appointment & Designation ---
    "tunjuk",      # NOTE: also in positive
    "angkat",      # NOTE: also in positive
}

# ═══════════════════════════════════════════════════════════════
# D. NEGATIVE FRAMING NOUNS (entity as TARGET of negative framing)
# ═══════════════════════════════════════════════════════════════
NEGATIVE_FRAMING_NOUNS = {
    # --- D1. Legal & Criminal ---
    "dugaan",      # allegation
    "terduga",     # suspect (adj/noun)
    "tersangka",   # suspect
    "tersangkut",  # implicated
    "dakwaan",     # indictment
    "vonis",       # verdict
    "hukuman",     # punishment
    "pidana",      # criminal (adj/noun)
    "tahanan",     # detainee
    "napi",        # prisoner
    "buron",       # fugitive
    "fugitive",    # loanword

    # --- D2. Corruption & Bribery ---
    "korupsi",     # corruption
    "suap",        # bribery
    "pungli",      # illegal levy
    "gratifikasi",  # gratification
    "markus",      # corruptor (slang)
    "pungguk",     # colloquial
    "koruptor",    # noun

    # --- D3. Scandal & Controversy ---
    "skandal",     # scandal
    "kontroversi",  # controversy
    "polemik",     # polemic
    "sensasi",     # sensation
    "skorsing",    # suspension
    "sanksi",      # sanction
    "teguran",     # reprimand

    # --- D4. Case & Lawsuit ---
    "kasus",       # case
    "perkara",     # case/lawsuit
    "tuntutan",    # demand/charge
    "gugatan",     # lawsuit
    "tuduhan",     # accusation
    "fitnah",      # slander
    "hujat",       # insult (noun)

    # --- D5. Violation & Deviation ---
    "pelanggaran",  # violation
    "penyimpangan",  # deviation
    "penyalahgunaan",  # misuse
    "pelanggar",   # violator
    "penyelewengan",  # deviation
    "abusif",      # abusive (loanword)

    # --- D6. Loss & Damage ---
    "rugi",        # loss
    "kerugian",    # loss (noun)
    "beban",       # burden
    "kehancuran",  # destruction
    "kemunduran",  # setback
    "kegagalan",   # failure
    "kerusakan",   # damage

    # --- D7. Evidence & Proof ---
    "bukti",       # evidence
    "ketahuan",    # caught
    "terbukti",    # proven
    "fakta",       # fact
    "rekam",       # record (noun)

    # --- D8. Removal & Dismissal ---
    "pemberhentian",  # dismissal
    "pencabutan",  # revocation
    "pemecatan",   # firing
    "pembubaran",  # dissolution
}

# ═══════════════════════════════════════════════════════════════
# E. POSITIVE FRAMING NOUNS (entity as TARGET of positive framing)
# ═══════════════════════════════════════════════════════════════
POSITIVE_FRAMING_NOUNS = {
    # --- E1. Praise & Recognition ---
    "pujian",      # praise
    "apresiasi",   # appreciation
    "sanjungan",   # adulation
    "penghargaan",  # award
    "pengakuan",   # recognition
    "hormat",      # respect
    "acungan",     # thumbs up

    # --- E2. Support & Endorsement ---
    "dukungan",    # support
    "restu",       # blessing
    "persetujuan",  # approval
    "dukungan",    # support (dup check)
    "backing",     # loanword
    "endorse",     # loanword

    # --- E3. Achievement & Success ---
    "prestasi",    # achievement
    "pencapaian",  # accomplishment
    "kesuksesan",  # success
    "sukses",      # success
    "kemenangan",  # victory
    "prestise",    # prestige
    "rekor",       # record

    # --- E4. Honor & Trust ---
    "kehormatan",  # honor
    "legitimasi",  # legitimacy
    "amanah",      # trust
    "mandat",      # mandate
    "kepercayaan",  # trust

    # --- E5. Quality & Excellence ---
    "unggul",      # excel
    "kualitas",    # quality
    "keunggulan",  # excellence
    "kompetensi",  # competence
    "kapabilitas",  # capability
}

# ═══════════════════════════════════════════════════════════════
# F. NEGATION WORDS (reverses sentiment polarity)
# ═══════════════════════════════════════════════════════════════
NEGATION_WORDS = {
    "tidak",       # not
    "bukan",       # not (copula)
    "jangan",      # don't
    "tak",         # not (formal)
    "tiada",       # no/none (formal)
    "tanpa",       # without
    "belom",       # belum (colloquial)
    "belum",       # not yet
    "nir",         # without (prefix)
    "anti",        # anti
    "kontra",      # contra
    "batal",       # cancel
    "urung",       # abort
}

# ═══════════════════════════════════════════════════════════════
# G. INTENSITY MODIFIERS (affects confidence level)
# ═══════════════════════════════════════════════════════════════
INTENSITY_HIGH = {
    "sangat",      # very
    "amat",        # very (formal)
    "betul",       # truly
    "sungguh",     # really
    "amat",        # very
    "begitu",      # so
    "terlalu",     # too
    "paling",      # most
    "ekstrem",     # extreme
    "parah",       # severe
}

INTENSITY_LOW = {
    "cukup",       # enough
    "agak",        # rather
    "lumayan",     # fairly
    "sedikit",     # a little
    "agaknya",     # somewhat
    "agak",        # rather
}

# ═══════════════════════════════════════════════════════════════
# H. HEDGING & CONDITIONAL MARKERS (reduces confidence)
# ═══════════════════════════════════════════════════════════════
HEDGING_WORDS = {
    "mungkin",         # maybe
    "barangkali",      # perhaps
    "sepertinya",      # it seems
    "kiranya",         # perhaps
    "agaknya",         # somewhat
    "kemungkinan",     # possibility
    "diduga",          # allegedly
    "diduga kuat",     # strongly suspected
    "konon",           # supposedly
    "katanya",         # they say
    "menurut",         # according to
    "dikabarkan",      # reported
    "diyakini",        # believed
    "disinyalir",      # signaled
}

CONDITIONAL_WORDS = {
    "jika",            # if
    "kalau",           # if (informal)
    "seandainya",      # if only
    "andai",           # if
    "seandainya",      # if only
    "bilamana",        # when/if (formal)
    "apabila",         # if (formal)
    "bila",            # if
    "asalkan",         # as long as
    "kecuali",         # unless
}

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
PRONOUNS = {"dia", "ia", "beliau", "mereka", "nya"}
QUOTE_CHARS = set('""""''')
MIN_LOCAL_CLAUSE_WORDS = 4
CLAUSE_SPLIT_RE = re.compile(
    r',|\byang\b|\bdan\b|\bsementara\b|\bsedangkan\b|\bnamun\b|\btetapi\b|\bsedang\b'
    r'|\bsoal\b|\btentang\b|\bterkait\b|\bmengenai\b|\bperihal\b',
    re.IGNORECASE,
)

# v20: TOKEN-OPTIMIZED — target 90% utilization
MAX_CONTEXT_WORDS = 160
MAX_CONTEXT_CHARS = 850
CONTEXT_WINDOW_SENTENCES = 3
DEFAULT_DAYS_BACK = 30

CONTEXT_VERSION = "v20_comprehensive"
