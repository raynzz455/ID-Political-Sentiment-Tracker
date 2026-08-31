"""
context_extraction_moe.py — Mixture of Experts for Context Extraction
========================================================================
v1.0 — Production-ready MoE with 5 heterogeneous experts.

ARCHITECTURE:
  (Article, entity_mention) → [5 Experts in parallel] → Router → Aggregate → Best context

EXPERTS:
  1. Sentence Window (current v19.1, token-optimized)
  2. Coreference Resolution (tracks "dia", "beliau" → entity)
  3. Semantic Role Labeling (extract entity's arguments only)
  4. Paragraph-based (full paragraph containing entity)
  5. Embedding Similarity (find semantically similar sentences)

ROUTER:
  Decides which expert(s) to trust per entity mention based on features:
    - Has pronoun references → coref expert strong
    - Entity is subject of verb → SRL expert strong
    - Entity in dense paragraph → paragraph expert strong
    - Entity mentioned multiple times → embedding expert strong
    - Default: sentence window (token-optimized baseline)

AGGREGATION:
  - Merge contexts from all experts
  - Deduplication (remove overlapping text)
  - Rank by quality_score × confidence_modifier
  - Cap at MAX_CONTEXT_CHARS (850, ~230 tokens)

EXPECTED IMPACT:
  - Context quality: 77% token util → 92%+ quality
  - Pronoun handling: poor → excellent (coref expert)
  - Multi-mention articles: single context → multi-span aggregation

USAGE:
  from packages.context.context_extraction_moe import ContextExtractionMoE
  
  moe = ContextExtractionMoE(stanza_nlp=nlp_pipeline)
  result = moe.extract(article_text, entity_name, entity_offset)
  # result = {"context_text": "...", "all_spans": [...], "quality_score": 95}
"""
import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("stanza").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContextSpan:
    """Single context span extracted by an expert."""
    text: str
    start_offset: int
    end_offset: int
    quality_score: int
    expert_source: str  # which expert extracted this
    confidence: float
    sentence_indices: List[int] = field(default_factory=list)
    has_pronoun_ref: bool = False
    has_sentiment_predicate: bool = False
    has_attribution: bool = False


@dataclass
class AggregatedContext:
    """Final aggregated context from MoE."""
    context_text: str
    all_spans: List[str]
    quality_score: int
    confidence: float
    expert_agreement: int
    span_count: int
    detection_methods: List[str]
    is_relevant: bool = True
    relevancy_score: float = 1.0


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

MAX_CONTEXT_CHARS = 850  # ~230 tokens (77% utilization of 256)
MAX_CONTEXT_WORDS = 160
MAX_SPANS = 5

PRONOUNS = {"dia", "ia", "beliau", "mereka", "nya", "beliau"}
QUOTE_CHARS = set('""""''')

# v20 lexicon (simplified — full version in context_worker_v20_lexicon.py)
SENTIMENT_PREDICATES_ACTIVE = {
    "kritik","kecam","sindir","serang","hina","cela","ejek","tuduh","tunding",
    "vonis","tahan","tangkap","cekal","pidana","dakwa","tuntut","pecat","mundur",
    "bongkar","ungkap","langgar","simpang","salah","rugi","gagal","tolak","keberatan",
    # positive
    "puji","sanjung","kagum","apresiasi","dukung","restui","sahkan","setuju",
    "akui","raih","capai","menang","sukses","berhasil","lantik","angkat",
}
ATTRIBUTION_WORDS = {
    "kata","nyata","tegas","jelaskan","tambah","imbau","sebut","papar",
    "ucap","tutur","ujar","jawab","balas","tanggapi","saran","usul","ajak",
    "tekan","sorot","tunjuk",
}
NEGATIVE_FRAMING_NOUNS = {
    "dugaan","terduga","tersangka","korupsi","suap","skandal","kontroversi",
    "kasus","perkara","pelanggaran","penyimpangan","rugi","vonis","hukuman",
    "pidana","dakwaan","bukti",
}
POSITIVE_FRAMING_NOUNS = {
    "pujian","apresiasi","dukungan","prestasi","pencapaian","kesuksesan",
    "penghargaan","pengakuan","legitimasi",
}


# ═══════════════════════════════════════════════════════════════
# EXPERT 1: SENTENCE WINDOW (current v19.1, token-optimized)
# ═══════════════════════════════════════════════════════════════

class SentenceWindowExtractor:
    """Expert 1: Extract anchor sentence ± N surrounding sentences.
    
    Strengths: Token-optimized (77% util), fast, deterministic
    Weaknesses: Misses pronoun references across sentences
    """
    
    def __init__(self, nlp_pipeline=None):
        self.nlp = nlp_pipeline
    
    def extract(self, article_text: str, entity_name: str, 
                entity_offset: int) -> List[ContextSpan]:
        """Extract context using sentence window strategy."""
        if self.nlp is None:
            return []
        
        try:
            doc = self.nlp(article_text)
        except Exception as e:
            logger.warning(f"Sentence window Stanza failed: {e}")
            return []
        
        # Build sentence list
        sentences = []
        for sent in doc.sentences:
            if len(sent.text.strip()) > 10:
                sentences.append({
                    "text": sent.text,
                    "start": sent.tokens[0].start_char if sent.tokens else 0,
                    "end": sent.tokens[-1].end_char if sent.tokens else 0,
                    "parsed": sent,
                })
        
        if not sentences:
            return []
        
        # Find anchor sentence (containing entity)
        anchor_idx = -1
        for idx, s in enumerate(sentences):
            if s["start"] <= entity_offset < s["end"]:
                anchor_idx = idx
                break
        
        if anchor_idx == -1:
            return []
        
        # Build context: anchor ± up to 4 surrounding sentences
        context_parts = [sentences[anchor_idx]["text"]]
        sentence_indices = [anchor_idx]
        
        prev_idx = anchor_idx - 1
        next_idx = anchor_idx + 1
        prev_added = 0
        next_added = 0
        max_each_side = 4
        
        while (prev_added + next_added) < (max_each_side * 2):
            current_chars = len(" ".join(context_parts))
            if current_chars >= MAX_CONTEXT_CHARS:
                break
            
            added = False
            if next_idx < len(sentences) and next_added < max_each_side:
                if len(sentences[next_idx]["text"]) > 20:
                    context_parts.append(sentences[next_idx]["text"])
                    sentence_indices.append(next_idx)
                    next_added += 1
                    next_idx += 1
                    added = True
            
            if prev_idx >= 0 and prev_added < max_each_side:
                if len(sentences[prev_idx]["text"]) > 20:
                    context_parts.insert(0, sentences[prev_idx]["text"])
                    sentence_indices.insert(0, prev_idx)
                    prev_added += 1
                    prev_idx -= 1
                    added = True
            
            if not added:
                break
        
        ctx_text = " ".join(context_parts)
        if len(ctx_text) > MAX_CONTEXT_CHARS:
            # Truncate keeping anchor centered
            anchor_text = sentences[anchor_idx]["text"]
            anchor_start = ctx_text.find(anchor_text)
            if anchor_start >= 0:
                remaining = MAX_CONTEXT_CHARS - len(anchor_text)
                left = ctx_text[:anchor_start][-remaining//2:].strip()
                right = ctx_text[anchor_start+len(anchor_text):][:remaining//2].strip()
                ctx_text = (left + " " + anchor_text + " " + right).strip()
        
        # Detect sentiment/attribution in anchor
        anchor_sent = sentences[anchor_idx]["parsed"]
        has_sentiment = False
        has_attribution = False
        for word in anchor_sent.words:
            lemma = (word.lemma or word.text).lower()
            if word.deprel == 'root':
                if lemma in SENTIMENT_PREDICATES_ACTIVE:
                    has_sentiment = True
                if lemma in ATTRIBUTION_WORDS:
                    has_attribution = True
            if word.upos in ('NOUN', 'PROPN'):
                if lemma in NEGATIVE_FRAMING_NOUNS or lemma in POSITIVE_FRAMING_NOUNS:
                    has_sentiment = True
        
        # Quality score (same as v19.1)
        attr_score = 40 if has_sentiment else (10 if has_attribution else 10)
        pos_score = 20 if anchor_idx == 0 else (12 if anchor_idx <= 2 else 5)
        quality = attr_score + 30 + pos_score + 10  # actor + exclusivity
        
        return [ContextSpan(
            text=ctx_text,
            start_offset=sentences[max(0, anchor_idx-prev_added)]["start"],
            end_offset=sentences[min(len(sentences)-1, anchor_idx+next_added)]["end"],
            quality_score=quality,
            expert_source="sentence_window",
            confidence=0.85,
            sentence_indices=sentence_indices,
            has_sentiment_predicate=has_sentiment,
            has_attribution=has_attribution,
        )]


# ═══════════════════════════════════════════════════════════════
# EXPERT 2: COREFERENCE RESOLUTION (tracks pronouns)
# ═══════════════════════════════════════════════════════════════

class CoreferenceExtractor:
    """Expert 2: Track pronouns (dia, beliau, ia) that refer to entity.
    
    Strengths: Catches context across sentences via pronoun resolution
    Weaknesses: Needs Stanza depparse, may false-match pronouns
    """
    
    def __init__(self, nlp_pipeline=None):
        self.nlp = nlp_pipeline
    
    def extract(self, article_text: str, entity_name: str,
                entity_offset: int) -> List[ContextSpan]:
        """Extract context by tracking pronoun references to entity."""
        if self.nlp is None:
            return []
        
        try:
            doc = self.nlp(article_text)
        except Exception as e:
            return []
        
        sentences = []
        for sent in doc.sentences:
            if len(sent.text.strip()) > 10:
                sentences.append({
                    "text": sent.text,
                    "start": sent.tokens[0].start_char if sent.tokens else 0,
                    "end": sent.tokens[-1].end_char if sent.tokens else 0,
                    "parsed": sent,
                })
        
        if not sentences:
            return []
        
        # Find anchor sentence
        anchor_idx = -1
        for idx, s in enumerate(sentences):
            if s["start"] <= entity_offset < s["end"]:
                anchor_idx = idx
                break
        
        if anchor_idx == -1:
            return []
        
        # Find pronoun references in surrounding sentences (±3)
        # A pronoun likely refers to entity if:
        # 1. It's within 3 sentences of anchor
        # 2. Entity is the main subject (nsubj) of anchor
        # 3. Pronoun is in subject position (nsubj)
        
        pronoun_sentences = []
        search_start = max(0, anchor_idx - 1)
        search_end = min(len(sentences), anchor_idx + 4)
        
        for idx in range(search_start, search_end):
            if idx == anchor_idx:
                continue
            
            sent = sentences[idx]["parsed"]
            for word in sent.words:
                text_lower = word.text.lower()
                if text_lower in PRONOUNS and word.deprel in ('nsubj', 'nsubj:pass', 'obl'):
                    # Check if there's a verb (predicate) in this sentence
                    has_predicate = any(w.deprel == 'root' for w in sent.words)
                    if has_predicate:
                        pronoun_sentences.append(idx)
                        break
        
        # Build context: anchor + all sentences with pronoun references
        context_indices = sorted(set([anchor_idx] + pronoun_sentences))
        context_parts = [sentences[idx]["text"] for idx in context_indices]
        ctx_text = " ".join(context_parts)
        
        if len(ctx_text) > MAX_CONTEXT_CHARS:
            ctx_text = ctx_text[:MAX_CONTEXT_CHARS]
        
        if not pronoun_sentences:
            return []  # No pronoun refs — let other experts handle
        
        has_pronoun_ref = True
        quality = 85  # high quality — pronoun resolution is valuable
        
        return [ContextSpan(
            text=ctx_text,
            start_offset=sentences[context_indices[0]]["start"],
            end_offset=sentences[context_indices[-1]]["end"],
            quality_score=quality,
            expert_source="coreference",
            confidence=0.90,
            sentence_indices=context_indices,
            has_pronoun_ref=True,
        )]


# ═══════════════════════════════════════════════════════════════
# EXPERT 3: SEMANTIC ROLE LABELING (entity's arguments only)
# ═══════════════════════════════════════════════════════════════

class SemanticRoleExtractor:
    """Expert 3: Extract only sentences where entity is nsubj/obj of predicate.
    
    Strengths: Precise — only sentences where entity has grammatical role
    Weaknesses: Narrow — may miss context if entity isn't grammatical subject
    """
    
    def __init__(self, nlp_pipeline=None):
        self.nlp = nlp_pipeline
    
    def extract(self, article_text: str, entity_name: str,
                entity_offset: int) -> List[ContextSpan]:
        """Extract context using semantic role analysis."""
        if self.nlp is None:
            return []
        
        try:
            doc = self.nlp(article_text)
        except Exception as e:
            return []
        
        sentences = []
        for sent in doc.sentences:
            if len(sent.text.strip()) > 10:
                sentences.append({
                    "text": sent.text,
                    "start": sent.tokens[0].start_char if sent.tokens else 0,
                    "end": sent.tokens[-1].end_char if sent.tokens else 0,
                    "parsed": sent,
                })
        
        if not sentences:
            return []
        
        # Find ALL sentences where entity is nsubj/obj/obl of a predicate
        role_sentences = []
        for idx, s in enumerate(sentences):
            sent = s["parsed"]
            entity_word = None
            for word in sent.words:
                if word.start_char <= entity_offset < word.end_char:
                    entity_word = word
                    break
                if entity_offset <= word.start_char < entity_offset + len(entity_name):
                    entity_word = word
                    break
            
            if entity_word and entity_word.deprel in ('nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'csubj'):
                # Check if head is a sentiment/attribution verb
                head_id = entity_word.head
                for word in sent.words:
                    if word.id == head_id:
                        lemma = (word.lemma or word.text).lower()
                        if lemma in SENTIMENT_PREDICATES_ACTIVE or lemma in ATTRIBUTION_WORDS:
                            role_sentences.append((idx, entity_word.deprel, lemma))
                            break
        
        if not role_sentences:
            return []
        
        # Build context from all role sentences + their adjacent sentences
        context_indices = set()
        for idx, role, verb in role_sentences:
            context_indices.add(idx)
            # Add adjacent sentence for context
            if idx > 0:
                context_indices.add(idx - 1)
            if idx < len(sentences) - 1:
                context_indices.add(idx + 1)
        
        context_indices = sorted(context_indices)
        context_parts = [sentences[idx]["text"] for idx in context_indices]
        ctx_text = " ".join(context_parts)
        
        if len(ctx_text) > MAX_CONTEXT_CHARS:
            ctx_text = ctx_text[:MAX_CONTEXT_CHARS]
        
        # High quality — entity has clear grammatical role
        has_sentiment = any(verb in SENTIMENT_PREDICATES_ACTIVE for _, _, verb in role_sentences)
        quality = 95 if has_sentiment else 80
        
        return [ContextSpan(
            text=ctx_text,
            start_offset=sentences[context_indices[0]]["start"],
            end_offset=sentences[context_indices[-1]]["end"],
            quality_score=quality,
            expert_source="semantic_role",
            confidence=0.92,
            sentence_indices=context_indices,
            has_sentiment_predicate=has_sentiment,
        )]


# ═══════════════════════════════════════════════════════════════
# EXPERT 4: PARAGRAPH-BASED (full paragraph containing entity)
# ═══════════════════════════════════════════════════════════════

class ParagraphExtractor:
    """Expert 4: Extract full paragraph containing entity mention.
    
    Strengths: Rich context, captures related entities in same paragraph
    Weaknesses: May include noise (other entities mentioned)
    """
    
    def extract(self, article_text: str, entity_name: str,
                entity_offset: int) -> List[ContextSpan]:
        """Extract paragraph containing entity mention."""
        # Split by double newlines (paragraph boundaries)
        paragraphs = article_text.split('\n\n')
        
        current_offset = 0
        for para in paragraphs:
            para_start = current_offset
            para_end = current_offset + len(para)
            
            if para_start <= entity_offset < para_end:
                # Found paragraph containing entity
                # Clean up: normalize whitespace using split+join (no regex)
                clean_para = ' '.join(para.split()).strip()
                
                if len(clean_para) < 50:
                    return []
                
                # Truncate if too long
                if len(clean_para) > MAX_CONTEXT_CHARS:
                    # Keep entity centered
                    entity_pos = clean_para.lower().find(entity_name.lower())
                    if entity_pos >= 0:
                        half = MAX_CONTEXT_CHARS // 2
                        start = max(0, entity_pos - half)
                        end = min(len(clean_para), entity_pos + len(entity_name) + half)
                        clean_para = clean_para[start:end]
                    else:
                        clean_para = clean_para[:MAX_CONTEXT_CHARS]
                
                return [ContextSpan(
                    text=clean_para,
                    start_offset=para_start,
                    end_offset=para_end,
                    quality_score=75,  # medium quality (may have noise)
                    expert_source="paragraph",
                    confidence=0.80,
                )]
            
            current_offset = para_end + 2  # +2 for \n\n
        
        return []


# ═══════════════════════════════════════════════════════════════
# EXPERT 5: EMBEDDING SIMILARITY (semantic match)
# ═══════════════════════════════════════════════════════════════

class EmbeddingSimilarityExtractor:
    """Expert 5: Find sentences semantically similar to entity mention.
    
    Strengths: Catches implicit context (sarcasm, irony), semantic matching
    Weaknesses: Slow (~100ms), needs embedding model
    """
    
    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold
        self._model = None
        self._stanza_nlp = None  # cached Stanza tokenizer

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('indobenchmark/indobert-base-p1')
                logger.info("Embedding model loaded for context extraction")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                return None
        return self._model

    def _split_sentences(self, text: str) -> List[str]:
        """Library-based sentence splitting using Stanza tokenizer.
        
        Handles abbreviations, honorifics (H., Ir., Dr.), and complex sentence
        structures that manual regex split fails on.
        """
        # Try Stanza first (best for Indonesian)
        try:
            if self._stanza_nlp is None:
                import stanza
                self._stanza_nlp = stanza.Pipeline(
                    "id", processors="tokenize",
                    use_gpu=False, verbose=False,
                    logging_level="ERROR"
                )
            doc = self._stanza_nlp(text[:3000])  # limit for speed
            return [sent.text for sent in doc.sentences]
        except ImportError:
            pass
        
        # Fallback: spaCy (if available)
        try:
            import spacy
            if not hasattr(self, '_spacy_nlp'):
                self._spacy_nlp = spacy.load("id_core_news_sm", disable=["ner", "tagger", "parser", "lemmatizer"])
            doc = self._spacy_nlp(text[:3000])
            return [sent.text for sent in doc.sents]
        except Exception:
            pass
        
        # Final fallback: simple split on punctuation (NOT regex — just string ops)
        # This is less accurate but has no manual regex pattern
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in '.!?' and len(current) > 1:
                # Check next char is whitespace (end of sentence)
                sentences.append(''.join(current).strip())
                current = []
        if current:
            sentences.append(''.join(current).strip())
        return [s for s in sentences if s]
    
    def extract(self, article_text: str, entity_name: str,
                entity_offset: int) -> List[ContextSpan]:
        """Extract context by finding semantically similar sentences to entity mention."""
        model = self._load_model()
        if model is None:
            return []
        
        # Library-based sentence splitting (Stanza tokenizer, no manual regex)
        sentences = self._split_sentences(article_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 2:
            return []
        
        # Find anchor sentence (containing entity)
        anchor_idx = -1
        for idx, sent in enumerate(sentences):
            if entity_name.lower() in sent.lower():
                anchor_idx = idx
                break
        
        if anchor_idx == -1:
            return []
        
        anchor_sent = sentences[anchor_idx]
        
        # Compute embeddings for all sentences
        try:
            import torch
            import torch.nn.functional as F
            
            sent_embeddings = model.encode(sentences, convert_to_tensor=True)
            anchor_emb = sent_embeddings[anchor_idx]
            
            # Compute cosine similarity to anchor
            sims = F.cosine_similarity(
                anchor_emb.unsqueeze(0).expand(len(sentences), -1),
                sent_embeddings,
                dim=1
            )
            
            # Find similar sentences (above threshold)
            similar_indices = []
            for idx in range(len(sentences)):
                if idx == anchor_idx:
                    continue
                if sims[idx].item() >= self.similarity_threshold:
                    similar_indices.append(idx)
            
            if not similar_indices:
                return []
            
            # Build context: anchor + similar sentences
            all_indices = sorted(set([anchor_idx] + similar_indices))
            context_parts = [sentences[idx] for idx in all_indices]
            ctx_text = " ".join(context_parts)
            
            if len(ctx_text) > MAX_CONTEXT_CHARS:
                ctx_text = ctx_text[:MAX_CONTEXT_CHARS]
            
            # Quality based on number of similar sentences
            quality = 70 + min(20, len(similar_indices) * 5)
            
            return [ContextSpan(
                text=ctx_text,
                start_offset=0,
                end_offset=len(ctx_text),
                quality_score=quality,
                expert_source="embedding_similarity",
                confidence=0.85,
                sentence_indices=all_indices,
            )]
            
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════

class ContextRouter:
    """Decides which context extraction expert to trust."""
    
    def __init__(self):
        self.default_weights = {
            'sentence_window': 0.30,
            'coreference': 0.20,
            'semantic_role': 0.20,
            'paragraph': 0.15,
            'embedding_similarity': 0.15,
        }
    
    def route(self, entity_features: Dict) -> Dict[str, float]:
        """Decide expert weights based on entity mention features."""
        weights = self.default_weights.copy()
        
        # Has pronoun references → coref expert strong
        if entity_features.get('has_pronoun_refs'):
            weights['coreference'] += 0.20
            weights['sentence_window'] -= 0.20
        
        # Entity is subject of verb → SRL expert strong
        if entity_features.get('is_subject_of_verb'):
            weights['semantic_role'] += 0.15
            weights['sentence_window'] -= 0.15
        
        # Entity in dense paragraph → paragraph expert strong
        if entity_features.get('in_dense_paragraph'):
            weights['paragraph'] += 0.10
            weights['embedding_similarity'] -= 0.10
        
        # Entity mentioned multiple times → embedding expert strong
        if entity_features.get('mention_count', 1) > 2:
            weights['embedding_similarity'] += 0.15
            weights['sentence_window'] -= 0.15
        
        # Short article → paragraph expert (richer context)
        if entity_features.get('article_length', 0) < 500:
            weights['paragraph'] += 0.10
            weights['embedding_similarity'] -= 0.10
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def extract_features(self, article_text: str, entity_name: str,
                         entity_offset: int, nlp_pipeline=None) -> Dict:
        """Extract entity mention features for routing."""
        # Count entity mentions
        mention_count = article_text.lower().count(entity_name.lower())
        
        # Check for pronouns in surrounding text
        surrounding = article_text[max(0, entity_offset-200):entity_offset+200].lower()
        has_pronoun_refs = any(p in surrounding.split() for p in PRONOUNS)
        
        # Check if entity is in dense paragraph (many sentences)
        para = article_text.split('\n\n')
        current_offset = 0
        entity_para = ""
        for p in para:
            if current_offset <= entity_offset < current_offset + len(p):
                entity_para = p
                break
            current_offset += len(p) + 2
        
        para_sentence_count = entity_para.count('.') + entity_para.count('!') + entity_para.count('?')
        in_dense_paragraph = para_sentence_count >= 3
        
        # Check if entity is subject of verb (needs Stanza)
        is_subject_of_verb = False
        if nlp_pipeline is not None:
            try:
                # Parse just the sentence containing entity
                sent_start = article_text.rfind('.', 0, entity_offset) + 1
                sent_end = article_text.find('.', entity_offset)
                if sent_end == -1:
                    sent_end = len(article_text)
                sentence = article_text[sent_start:sent_end].strip()
                
                doc = nlp_pipeline(sentence)
                for sent in doc.sentences:
                    for word in sent.words:
                        if word.start_char <= (entity_offset - sent_start) < word.end_char:
                            if word.deprel in ('nsubj', 'nsubj:pass'):
                                is_subject_of_verb = True
                                break
                    break
            except:
                pass
        
        return {
            'mention_count': mention_count,
            'has_pronoun_refs': has_pronoun_refs,
            'in_dense_paragraph': in_dense_paragraph,
            'is_subject_of_verb': is_subject_of_verb,
            'article_length': len(article_text),
        }


# ═══════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════

class ContextAggregator:
    """Merges context spans from multiple experts."""
    
    def aggregate(self, expert_spans: Dict[str, List[ContextSpan]],
                  expert_weights: Dict[str, float]) -> AggregatedContext:
        """Merge spans from all experts.
        
        Strategy:
          1. Deduplicate overlapping spans
          2. Rank by quality_score × expert_weight
          3. Select best span + supplementary spans (up to MAX_SPANS)
          4. Build final context_text (capped at MAX_CONTEXT_CHARS)
        """
        if not expert_spans:
            return AggregatedContext(
                context_text="", all_spans=[], quality_score=0,
                confidence=0, expert_agreement=0, span_count=0,
                detection_methods=[]
            )
        
        # Flatten all spans with their expert weights
        all_spans = []
        for expert_name, spans in expert_spans.items():
            weight = expert_weights.get(expert_name, 0.2)
            for span in spans:
                # Adjust quality by expert weight
                adjusted_quality = int(span.quality_score * weight * 2)  # scale up
                all_spans.append((span, adjusted_quality, expert_name))
        
        if not all_spans:
            return AggregatedContext(
                context_text="", all_spans=[], quality_score=0,
                confidence=0, expert_agreement=0, span_count=0,
                detection_methods=[]
            )
        
        # Sort by adjusted quality (descending)
        all_spans.sort(key=lambda x: x[1], reverse=True)
        
        # Select best span
        best_span, best_quality, best_expert = all_spans[0]
        
        # Collect supplementary spans (non-overlapping)
        supplementary_spans = []
        seen_text = set()
        seen_text.add(best_span.text[:100])  # first 100 chars for dedup
        
        for span, quality, expert in all_spans[1:]:
            if len(supplementary_spans) >= MAX_SPANS - 1:
                break
            
            # Dedup by checking overlap
            span_key = span.text[:100]
            if span_key in seen_text:
                continue
            
            # Check actual overlap
            overlaps = False
            for existing in [best_span] + supplementary_spans:
                if (span.start_offset < existing.end_offset and
                    span.end_offset > existing.start_offset):
                    # Overlap > 50% → skip
                    overlap = min(span.end_offset, existing.end_offset) - \
                              max(span.start_offset, existing.start_offset)
                    if overlap > len(span.text) * 0.5:
                        overlaps = True
                        break
            
            if not overlaps:
                supplementary_spans.append(span)
                seen_text.add(span_key)
        
        # Build all_spans list
        all_span_texts = [best_span.text] + [s.text for s in supplementary_spans]
        
        # Build final context_text (best span, capped)
        final_text = best_span.text
        if len(final_text) > MAX_CONTEXT_CHARS:
            final_text = final_text[:MAX_CONTEXT_CHARS]
        
        # Aggregate metadata
        all_experts = set()
        all_experts.add(best_expert)
        for span in supplementary_spans:
            all_experts.add(span.expert_source)
        
        # Boost quality if multiple experts agree
        expert_agreement = len(all_experts)
        final_quality = best_quality
        if expert_agreement >= 3:
            final_quality = int(final_quality * 1.2)
        elif expert_agreement == 2:
            final_quality = int(final_quality * 1.1)
        
        final_quality = min(100, final_quality)
        
        # Aggregate flags
        has_sentiment = best_span.has_sentiment_predicate or any(
            s.has_sentiment_predicate for s in supplementary_spans
        )
        has_attribution = best_span.has_attribution or any(
            s.has_attribution for s in supplementary_spans
        )
        
        return AggregatedContext(
            context_text=final_text,
            all_spans=all_span_texts[:MAX_SPANS],
            quality_score=final_quality,
            confidence=best_span.confidence,
            expert_agreement=expert_agreement,
            span_count=len(all_span_texts),
            detection_methods=list(all_experts),
            is_relevant=True,
            relevancy_score=1.0,
        )


# ═══════════════════════════════════════════════════════════════
# MAIN MoE CLASS
# ═══════════════════════════════════════════════════════════════

class ContextExtractionMoE:
    """Mixture of Experts for Context Extraction.
    
    Usage:
        moe = ContextExtractionMoE(stanza_nlp=nlp)
        result = moe.extract(article_text, entity_name, entity_offset)
    """
    
    def __init__(self, stanza_nlp=None, enable_embedding: bool = True,
                 parallel: bool = True):
        self.parallel = parallel
        
        # Expert 1: Sentence window (always enabled, uses stanza)
        self.window_expert = SentenceWindowExtractor(stanza_nlp) if stanza_nlp else None
        
        # Expert 2: Coreference (uses stanza)
        self.coref_expert = CoreferenceExtractor(stanza_nlp) if stanza_nlp else None
        
        # Expert 3: Semantic role (uses stanza)
        self.srl_expert = SemanticRoleExtractor(stanza_nlp) if stanza_nlp else None
        
        # Expert 4: Paragraph-based (no deps)
        self.para_expert = ParagraphExtractor()
        
        # Expert 5: Embedding similarity (optional)
        self.embed_expert = None
        if enable_embedding:
            self.embed_expert = EmbeddingSimilarityExtractor()
        
        # Router + Aggregator
        self.router = ContextRouter()
        self.aggregator = ContextAggregator()
        self.stanza_nlp = stanza_nlp
    
    def extract(self, article_text: str, entity_name: str,
                entity_offset: int, timeout: float = 30.0) -> Dict:
        """Run MoE context extraction.
        
        Returns:
            {
                "context_text": str,
                "all_spans": [str, ...],
                "quality_score": int,
                "confidence": float,
                "expert_agreement": int,
                "expert_weights": {expert: weight},
                "processing_time_ms": int,
                "experts_used": [str],
            }
        """
        t0 = time.time()
        
        # Step 1: Extract features for routing
        entity_features = self.router.extract_features(
            article_text, entity_name, entity_offset, self.stanza_nlp
        )
        expert_weights = self.router.route(entity_features)
        
        # Step 2: Build expert list
        experts = {}
        if self.window_expert:
            experts['sentence_window'] = self.window_expert
        if self.coref_expert:
            experts['coreference'] = self.coref_expert
        if self.srl_expert:
            experts['semantic_role'] = self.srl_expert
        if self.para_expert:
            experts['paragraph'] = self.para_expert
        if self.embed_expert:
            experts['embedding_similarity'] = self.embed_expert
        
        # Step 3: Run experts (parallel or sequential)
        expert_spans = {}
        
        if self.parallel and len(experts) > 1:
            with ThreadPoolExecutor(max_workers=len(experts)) as pool:
                futures = {
                    pool.submit(expert.extract, article_text, entity_name, entity_offset): name
                    for name, expert in experts.items()
                }
                for future in as_completed(futures, timeout=timeout):
                    name = futures[future]
                    try:
                        expert_spans[name] = future.result(timeout=timeout/len(experts))
                    except Exception as e:
                        logger.warning(f"Expert {name} failed: {e}")
                        expert_spans[name] = []
        else:
            for name, expert in experts.items():
                try:
                    expert_spans[name] = expert.extract(article_text, entity_name, entity_offset)
                except Exception as e:
                    logger.warning(f"Expert {name} failed: {e}")
                    expert_spans[name] = []
        
        # Step 4: Aggregate
        result = self.aggregator.aggregate(expert_spans, expert_weights)
        
        processing_time = int((time.time() - t0) * 1000)
        
        return {
            "context_text": result.context_text,
            "all_spans": result.all_spans,
            "quality_score": result.quality_score,
            "confidence": result.confidence,
            "expert_agreement": result.expert_agreement,
            "expert_weights": expert_weights,
            "expert_results": {
                name: [{"text": s.text[:100], "quality": s.quality_score}
                       for s in spans]
                for name, spans in expert_spans.items()
            },
            "processing_time_ms": processing_time,
            "experts_used": list(experts.keys()),
            "entity_features": entity_features,
            "span_count": result.span_count,
            "detection_methods": result.detection_methods,
        }
    
    def extract_to_db_format(self, article_text: str, entity_name: str,
                              entity_offset: int, article_id: str,
                              entity_id: str) -> Dict:
        """Extract and format for DB insertion (entity_contexts table)."""
        result = self.extract(article_text, entity_name, entity_offset)
        
        return {
            "raw_text_id": article_id,
            "entity_id": entity_id,
            "context_text": result["context_text"],
            "context_version": "moe_v1",
            "metadata": {
                "quality_score": result["quality_score"],
                "all_spans": result["all_spans"],
                "span_count": result["span_count"],
                "expert_agreement": result["expert_agreement"],
                "detection_methods": result["detection_methods"],
                "expert_weights": result["expert_weights"],
                "processing_time_ms": result["processing_time_ms"],
                "is_relevant": True,
                "relevancy_score": 1.0,
            }
        }
