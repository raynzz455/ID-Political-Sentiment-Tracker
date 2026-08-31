"""
enricher_worker_library.py
==========================
Library-based enricher worker (no manual regex/word lists).

Uses:
  - KeyBERT for keyword extraction (embedding-based)
  - BERTopic for topic modeling (transformer-based)
  - transformers pipeline for emotion classification (IndoBERT)
  - Stanza for POS/lemma (no manual tagging)

Replaces old manual enricher that used:
  - Hardcoded keyword lists
  - Manual regex patterns for entity extraction
  - Rule-based emotion detection
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)


@dataclass
class EnrichedArticle:
    """Result of enrichment pipeline."""
    article_id: str
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    emotion: str = ""
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    formality_score: float = 0.0  # 0-1 (1=formal, 0=informal)
    entities_mentioned: List[str] = field(default_factory=list)
    summary: str = ""


class LibraryEnricher:
    """Enricher worker using NLP libraries (no manual regex)."""

    def __init__(self,
                 enable_keywords: bool = True,
                 enable_topics: bool = False,  # BERTopic is heavy
                 enable_emotion: bool = True,
                 enable_formality: bool = True):
        self.enable_keywords = enable_keywords
        self.enable_topics = enable_topics
        self.enable_emotion = enable_emotion
        self.enable_formality = enable_formality

        # Lazy-loaded models (only initialize when first used)
        self._kw_model = None       # KeyBERT
        self._topic_model = None   # BERTopic
        self._emotion_pipe = None   # transformers pipeline
        self._stanza_nlp = None     # Stanza for POS/formality

    # ═══════════════════════════════════════════════════════════════
    # MODEL LOADERS (lazy init)
    # ═══════════════════════════════════════════════════════════════

    def _load_keyword_model(self):
        """Load KeyBERT for keyword extraction."""
        if self._kw_model is None:
            try:
                from keybert import KeyBERT
                # Uses sentence-transformers under the hood
                self._kw_model = KeyBERT('indobenchmark/indobert-base-p1')
                logger.info("KeyBERT model loaded")
            except ImportError:
                logger.warning("KeyBERT not installed. Run: pip install keybert")
                return None
            except Exception as e:
                logger.warning(f"Failed to load KeyBERT: {e}")
                return None
        return self._kw_model

    def _load_topic_model(self):
        """Load BERTopic for topic modeling."""
        if self._topic_model is None:
            try:
                from bertopic import BERTopic
                self._topic_model = BERTopic(
                    language="multilingual",  # supports Indonesian
                    calculate_probabilities=False,
                    nr_topics=10,
                )
                logger.info("BERTopic model loaded")
            except ImportError:
                logger.warning("BERTopic not installed. Run: pip install bertopic")
                return None
            except Exception as e:
                logger.warning(f"Failed to load BERTopic: {e}")
                return None
        return self._topic_model

    def _load_emotion_model(self):
        """Load IndoBERT emotion classifier."""
        if self._emotion_pipe is None:
            try:
                from transformers import pipeline
                # Indonesian sentiment/emotion model
                self._emotion_pipe = pipeline(
                    "text-classification",
                    model="indobenchmark/indobert-base-p1",
                    return_all_scores=True,
                )
                logger.info("Emotion pipeline loaded")
            except ImportError:
                logger.warning("transformers not installed. Run: pip install transformers")
                return None
            except Exception as e:
                logger.warning(f"Failed to load emotion model: {e}")
                return None
        return self._emotion_pipe

    def _load_stanza(self):
        """Load Stanza for POS/formality analysis."""
        if self._stanza_nlp is None:
            try:
                import stanza
                self._stanza_nlp = stanza.Pipeline(
                    "id", processors="tokenize,pos",
                    use_gpu=False, verbose=False,
                    logging_level="ERROR"
                )
                logger.info("Stanza POS pipeline loaded")
            except ImportError:
                logger.warning("stanza not installed. Run: pip install stanza")
                return None
            except Exception as e:
                logger.warning(f"Failed to load Stanza: {e}")
                return None
        return self._stanza_nlp

    # ═══════════════════════════════════════════════════════════════
    # ENRICHMENT METHODS
    # ═══════════════════════════════════════════════════════════════

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords using KeyBERT (embedding-based, no manual word list)."""
        model = self._load_keyword_model()
        if model is None:
            return []
        try:
            # KeyBERT returns list of (keyword, score)
            keywords = model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),  # 1-2 word phrases
                stop_words=None,  # let model handle it (multilingual)
                top_n=top_k,
                use_mmr=True,  # Maximal Marginal Relevance for diversity
                diversity=0.7,
            )
            return [kw for kw, score in keywords if score > 0.3]
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def extract_topics(self, text: str) -> List[str]:
        """Extract topics using BERTopic (transformer-based clustering)."""
        model = self._load_topic_model()
        if model is None:
            return []
        try:
            # BERTopic works best on document collections, but can do single doc
            topics, probs = model.fit_transform([text])
            # Get topic representation (top words)
            topic_info = model.get_topic_info()
            if len(topic_info) > 0:
                topic_id = topics[0]
                if topic_id >= 0:
                    topic_words = model.get_topic(topic_id)
                    return [word for word, _ in topic_words[:5]]
            return []
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return []

    def classify_emotion(self, text: str) -> Dict[str, float]:
        """Classify emotion/sentiment using IndoBERT."""
        pipe = self._load_emotion_model()
        if pipe is None:
            return {}
        try:
            results = pipe(text[:512])  # truncate to model max length
            if results and isinstance(results[0], list):
                # return_all_scores=True returns list of dicts
                return {r['label']: r['score'] for r in results[0]}
            return {}
        except Exception as e:
            logger.warning(f"Emotion classification failed: {e}")
            return {}

    def compute_formality_score(self, text: str) -> float:
        """Compute formality score using Stanza POS distribution.

        Returns 0-1 where:
          1.0 = very formal (academic/legal/news)
          0.0 = very informal (colloquial/slang)

        Library-based: uses POS tag distribution, NOT hardcoded word lists.
        """
        nlp = self._load_stanza()
        if nlp is None:
            return 0.5  # default neutral

        try:
            doc = nlp(text[:2000])  # limit for speed
            pos_tags = [word.upos for sent in doc.sentences for word in sent.words]

            if not pos_tags:
                return 0.5

            # Formal indicators (high % in formal text)
            formal_tags = {"NOUN", "PROPN", "VERB", "ADJ", "ADP", "DET", "SCONJ"}
            formal_count = sum(1 for p in pos_tags if p in formal_tags)

            # Informal indicators (high % in informal text)
            informal_tags = {"INTJ", "PART", "ADV", "X"}  # X = unknown/foreign
            informal_count = sum(1 for p in pos_tags if p in informal_tags)

            total = len(pos_tags)
            formal_ratio = formal_count / total
            informal_ratio = informal_count / total

            # Normalize: formal text typically has formal_ratio > 0.6
            # informal text has informal_ratio > 0.2
            score = formal_ratio - informal_ratio
            # Clamp to 0-1
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Formality scoring failed: {e}")
            return 0.5

    def extract_entities_via_ner(self, text: str) -> List[str]:
        """Extract entity names using Stanza NER (library, not regex)."""
        nlp = self._load_stanza()
        if nlp is None:
            return []
        try:
            # Need NER processor
            if not hasattr(self, '_stanza_ner'):
                import stanza
                self._stanza_ner = stanza.Pipeline(
                    "id", processors="tokenize,ner",
                    use_gpu=False, verbose=False,
                    logging_level="ERROR"
                )
            doc = self._stanza_ner(text[:2000])
            entities = []
            for ent in doc.ents:
                if ent.type == "PER":  # Person entities
                    entities.append(ent.text)
            return list(set(entities))  # dedupe
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════
    # MAIN ENRICHMENT PIPELINE
    # ═══════════════════════════════════════════════════════════════

    def enrich(self, article_id: str, text: str) -> EnrichedArticle:
        """Run full enrichment pipeline on article."""
        result = EnrichedArticle(article_id=article_id)

        if not text or len(text.strip()) < 50:
            return result

        # Run all enrichments in parallel (when models loaded)
        if self.enable_keywords:
            result.keywords = self.extract_keywords(text)

        if self.enable_topics:
            result.topics = self.extract_topics(text)

        if self.enable_emotion:
            scores = self.classify_emotion(text)
            result.emotion_scores = scores
            if scores:
                result.emotion = max(scores, key=scores.get)

        if self.enable_formality:
            result.formality_score = self.compute_formality_score(text)

        # NER-based entity extraction (library, not regex)
        result.entities_mentioned = self.extract_entities_via_ner(text)

        return result


# ═══════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Install required:
    #   pip install keybert bertopic transformers stanza sentence-transformers torch

    enricher = LibraryEnricher(
        enable_keywords=True,
        enable_topics=False,  # BERTopic is heavy, disable for speed
        enable_emotion=True,
        enable_formality=True,
    )

    sample_text = """
    Presiden Prabowo Subianto melantik Erick Thohir sebagai Menteri Pemuda
    dan Olahraga di Istana Negara, Rabu. Pelantikan ini menggantikan Dito
    Ariotedjo yang dicopot pekan lalu. Erick Thohir sebelumnya menjabat
    sebagai Menteri BUMN.
    """

    result = enricher.enrich("test-001", sample_text)
    print(f"Keywords: {result.keywords}")
    print(f"Emotion: {result.emotion} (scores: {result.emotion_scores})")
    print(f"Formality: {result.formality_score:.2f}")
    print(f"Entities (NER): {result.entities_mentioned}")
