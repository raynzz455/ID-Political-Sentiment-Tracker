# Refactor Plan: Library-Based Workers
# ====================================
# Kritik user: hindari regex manual, gunakan library NLP yang compatible.

## STATUS SAAT INI

### Entity Resolution MoE (entity_resolution_moe.py)
- 6 experts: regex, stanza_ner, spacy_ner, dbpedia, embedding_fuzzy, polyglot
- ❌ "regex" expert masih pakai hardcoded pattern matching
- ❌ Feature extraction (formal/slang detection) pakai regex + word list manual
- ✅ Stanza, spaCy, DBpedia, sentence-transformers sudah dipakai

### Context Extraction MoE (context_extraction_moe.py)
- 5 experts: sentence_window, coreference, semantic_role, paragraph, embedding_similarity
- ❌ "sentence_window" expert masih pakai manual sentence splitting
- ❌ "paragraph" expert masih pakai manual paragraph detection
- ✅ coreference, SRL, embedding sudah pakai library

### Enricher Worker
- Belum ditemukan di repo lokal (mungkin di GitHub repo yang belum di-clone)
- Perlu di-review setelah clone

## REKOMENDASI REFACTOR

### 1. Entity Resolution Worker — Ganti Regex dengan Library

#### SEBELUM (regex manual):
```python
# Deteksi formal name
has_formal = bool(re.search(r'\b(H\.|Ir\.|Dr\.|Prof\.|KH\.|Haji)\s+...', text))
# Deteksi slang
slang_markers = ['cak', 'gus', 'mas', 'mbak', ...]
has_slang = any(m in text.lower() for m in slang_markers)
```

#### SESUDAH (library-based):

#### a) Formal name detection → pakai spaCy NER + POS
```python
import spacy
nlp = spacy.load("id_core_news_sm")  # atau xx_ent_wiki_sm
doc = nlp(text)
has_formal = any(ent.label_ == "PERSON" for ent in doc.ents)
# POS tagging untuk deteksi title + name
has_title_name = any(tok.text in {"H.", "Ir.", "Dr.", "Prof.", "KH.", "Haji"} 
                     and tok.nbor(1).pos_ == "PROPN" for tok in doc)
```

#### b) Slang/colloquial detection → pakai Stanza + language model
```python
import stanza
nlp = stanza.Pipeline("id", processors="tokenize,pos,lemma")
doc = nlp(text)
# Stanza POS: deteksi informal markers via POS distribution
# Formal text: high % NOUN/PROPN/VERB, low % INTJ/ADV
# Informal: high % INTJ, PART, slang ADV
pos_tags = [word.upos for sent in doc.sentences for word in sent.words]
formal_score = sum(1 for p in pos_tags if p in {"NOUN","PROPN","VERB","ADJ"}) / len(pos_tags)
has_slang = formal_score < 0.6  # threshold
```

#### c) Entity matching → pakai rapidfuzz (bukan regex)
```python
from rapidfuzz import fuzz, process
# Alias matching dengan fuzzy
matches = process.extract(query, entity_list, scorer=fuzz.WRatio, limit=5)
# Lebih cepat dari regex, handle typo, case-insensitive otomatis
```

### 2. Context Extraction Worker — Ganti Manual dengan Stanza

#### SEBELUM (manual sentence splitting):
```python
# Manual sentence split
sentences = re.split(r'(?<=[.!?])\s+', text)
# Manual paragraph detection
paragraphs = text.split('\n\n')
```

#### SESUDAH (Stanza sentence segmentation):
```python
import stanza
nlp = stanza.Pipeline("id", processors="tokenize")
doc = nlp(text)
sentences = [sent.text for sent in doc.sentences]  # Stanza handle abbreviations
# Paragraph: pakai doc.to_dict() atau sent para boundaries
```

#### SEBELUM (manual window):
```python
# Manual context window
start = max(0, entity_pos - 200)
end = min(len(text), entity_pos + 200)
context = text[start:end]
```

#### SESUDAH (dependency parsing untuk context):
```python
# Stanza dependency parsing — ambil kalimat yang terhubung
doc = stanza_nlp(text)
entity_sent_idx = find_sentence_with_entity(doc, entity)
# Ambil kalimat sebelum + current + sesudah (context yang benar)
context_sents = doc.sentences[max(0, entity_sent_idx-1):entity_sent_idx+2]
context = " ".join(s.text for s in context_sents)
```

### 3. Enricher Worker — Library-Based

#### Rekomendasi library untuk enricher:
```python
# Sentiment: sudah pakai IndoBERT (production)
# Emotion: pakai IndoBERT emotion model
from transformers import pipeline
emotion_classifier = pipeline("text-classification", model="indobenchmark/indobert-base-p1")

# Topic modeling: pakai BERTopic (bukan regex)
from bertopic import BERTopic
topic_model = BERTopic(language="indonesian", calculate_probabilities=True)

# Keyword extraction: pakai KeyBERT (bukan TF-IDF manual)
from keybert import KeyBERT
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2))

# NER tambahan: pakai Stanza + spaCy ensemble
# Entity linking: pakai DBpedia Spotlight (sudah dipakai)
```

## LIBRARY YANG DIREKOMENDASIKAN (compatible, Indonesian-support)

| Task | Library | Install | Keunggulan |
|------|---------|---------|------------|
| **NER** | stanza | `pip install stanza` | Indonesian model, SOTA |
| **NER alt** | spacy + id_core_news_sm | `pip install spacy && python -m spacy download id_core_news_sm` | Fast, lightweight |
| **NER alt** | polyglot | `pip install polyglot pyicu pycld2` | Multi-language |
| **Fuzzy matching** | rapidfuzz | `pip install rapidfuzz` | 10x faster than fuzzywuzzy |
| **Embedding** | sentence-transformers | `pip install sentence-transformers` | Multilingual, SOTA |
| **Sentence split** | stanza (tokenize) | (sudah install) | Handle abbreviations |
| **Coreference** | stanza + neuralcoref | `pip install neuralcoref` | Spanish/English, perlu adapt |
| **SRL** | allen-nlp | `pip install allennlp` | Complex setup |
| **Topic modeling** | BERTopic | `pip install bertopic` | Berbasis transformer |
| **Keyword** | KeyBERT | `pip install keybert` | Embedding-based |
| **Emotion** | transformers pipeline | `pip install transformers` | Pre-trained |
| **DBpedia** | spotlight | REST API | No install needed |

## PRIORITAS REFACTOR

1. **HIGH**: Hapus "regex" expert di entity MoE → ganti dengan Stanza+spaCy ensemble
2. **HIGH**: Hapus manual sentence split di context MoE → pakai Stanza tokenizer
3. **MEDIUM**: Hapus slang list manual → pakai Stanza POS distribution
4. **MEDIUM**: Tambah KeyBERT untuk keyword extraction di enricher
5. **LOW**: Tambah BERTopic untuk topic modeling di enricher

## CATATAN

- Library yang sudah dipakai (Stanza, spaCy, sentence-transformers, DBpedia) sudah benar
- Yang perlu dihapus: regex manual untuk text analysis (bukan untuk alias matching)
- Alias matching dengan `re.compile(r'\b' + re.escape(name) + r'\b')` itu OK — itu word boundary, bukan text analysis
- Fokus: ganti text analysis manual dengan library NLP
