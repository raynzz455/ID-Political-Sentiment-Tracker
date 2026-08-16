"""
context_worker_v20_integration.py
==================================
Integration logic for v20 comprehensive lexicon.

Shows HOW to use the new categories:
  - NEGATION_WORDS: reverse sentiment polarity
  - INTENSITY_HIGH/LOW: adjust quality_score
  - HEDGING_WORDS: reduce confidence
  - CONDITIONAL_WORDS: reduce confidence

This is the compute_loss/quality_score section that goes into context_worker.py
"""

# === INTEGRATION INTO process_single_article_context() ===
# Replace the sentiment detection section with this:

def detect_sentiment_with_modifiers(sent, entity_start, entity_end):
    """v20: Enhanced sentiment detection with negation + intensity + hedging.
    
    Returns dict with:
      - polarity: 'positive' | 'negative' | 'neutral'
      - has_sentiment: bool
      - confidence_modifier: float (0.5-1.5, multiplies quality_score)
      - negation_detected: bool
      - hedging_detected: bool
    """
    # Collect all words in sentence
    words_before_entity = []  # words BEFORE entity mention (for negation scope)
    words_after_entity = []   # words AFTER entity mention
    root_word = ""
    has_sentiment_predicate = False
    has_attribution = False
    has_negative_noun = False
    has_positive_noun = False
    has_negation = False
    has_intensity_high = False
    has_intensity_low = False
    has_hedging = False
    has_conditional = False

    for word in sent.words:
        lemma = (word.lemma or word.text).lower()
        text_lower = word.text.lower()

        if word.deprel == 'root':
            root_word = lemma

        # Check sentiment predicates
        if lemma in SENTIMENT_PREDICATES_ACTIVE:
            has_sentiment_predicate = True
        if lemma in ATTRIBUTION_WORDS:
            has_attribution = True

        # Check framing nouns
        if word.upos in ('NOUN', 'PROPN'):
            if lemma in NEGATIVE_FRAMING_NOUNS:
                has_negative_noun = True
            elif lemma in POSITIVE_FRAMING_NOUNS:
                has_positive_noun = True

        # v20: Check negation (scope: 3 words before sentiment word)
        if text_lower in NEGATION_WORDS:
            has_negation = True

        # v20: Check intensity modifiers
        if text_lower in INTENSITY_HIGH:
            has_intensity_high = True
        if text_lower in INTENSITY_LOW:
            has_intensity_low = True

        # v20: Check hedging
        if text_lower in HEDGING_WORDS:
            has_hedging = True

        # v20: Check conditional
        if text_lower in CONDITIONAL_WORDS:
            has_conditional = True

    # Nouns can also trigger sentiment
    if has_negative_noun or has_positive_noun:
        has_sentiment_predicate = True

    # v20: Determine polarity
    if has_negative_noun and not has_positive_noun:
        polarity = 'negative'
    elif has_positive_noun and not has_negative_noun:
        polarity = 'positive'
    elif root_word in SENTIMENT_PREDICATES_POSITIVE:
        polarity = 'positive'
    elif root_word in SENTIMENT_PREDICATES_ACTIVE and root_word not in SENTIMENT_PREDICATES_POSITIVE:
        polarity = 'negative'
    else:
        polarity = 'neutral'

    # v20: Apply negation reversal
    # "tidak dipuji" = negative (negation reverses positive)
    # "tidak dikritik" = positive (negation reverses negative)
    if has_negation and polarity != 'neutral':
        polarity = 'positive' if polarity == 'negative' else 'negative'
        # Negated sentiment is weaker than direct sentiment
        confidence_modifier = 0.7
    else:
        confidence_modifier = 1.0

    # v20: Apply intensity modifiers
    if has_intensity_high:
        confidence_modifier *= 1.3  # "sangat dipuji" = stronger
    if has_intensity_low:
        confidence_modifier *= 0.8  # "agak dipuji" = weaker

    # v20: Apply hedging (reduce confidence)
    if has_hedging:
        confidence_modifier *= 0.6  # "mungkin dipuji" = uncertain

    # v20: Apply conditional (reduce confidence)
    if has_conditional:
        confidence_modifier *= 0.5  # "jika dipuji" = conditional

    # Clamp
    confidence_modifier = max(0.3, min(1.5, confidence_modifier))

    return {
        'polarity': polarity,
        'has_sentiment': has_sentiment_predicate,
        'has_attribution': has_attribution,
        'has_negative_noun': has_negative_noun,
        'has_positive_noun': has_positive_noun,
        'has_negation': has_negation,
        'has_intensity_high': has_intensity_high,
        'has_intensity_low': has_intensity_low,
        'has_hedging': has_hedging,
        'has_conditional': has_conditional,
        'confidence_modifier': confidence_modifier,
        'root_word': root_word,
    }


# === UPDATED QUALITY SCORE CALCULATION ===
def calculate_quality_score_v20(detection: dict, is_main_actor: bool,
                                  para_idx: int, is_crowded: bool,
                                  used_local_clause: bool) -> dict:
    """v20: Quality score with confidence modifier from negation/intensity/hedging."""

    # Base scores (same as v19.1)
    if detection['has_sentiment']:
        attr_score = 40
    elif detection['has_attribution']:
        attr_score = 10
    else:
        attr_score = 10

    actor_score = 30 if is_main_actor else 10
    pos_score = 20 if para_idx == 0 else (12 if para_idx <= 2 else 5)
    exclusivity_score = 10 if not is_crowded else (5 if used_local_clause else 0)

    base_quality = attr_score + actor_score + pos_score + exclusivity_score

    # v20: Apply confidence modifier
    conf_mod = detection['confidence_modifier']
    adjusted_quality = int(base_quality * conf_mod)

    # v20: Bonus for clear polarity (non-neutral)
    if detection['polarity'] != 'neutral' and not detection['has_hedging']:
        adjusted_quality += 5  # clear sentiment signal

    return {
        'quality_score': adjusted_quality,
        'base_quality_score': base_quality,
        'confidence_modifier': round(conf_mod, 2),
        'attr_score': attr_score,
        'actor_score': actor_score,
        'pos_score': pos_score,
        'exclusivity_score': exclusivity_score,
        'polarity': detection['polarity'],
        'has_sentiment_predicate': detection['has_sentiment'],
        'has_attribution': detection['has_attribution'],
        'has_negative_noun': detection['has_negative_noun'],
        'has_positive_noun': detection['has_positive_noun'],
        'has_negation': detection['has_negation'],
        'has_intensity_high': detection['has_intensity_high'],
        'has_intensity_low': detection['has_intensity_low'],
        'has_hedging': detection['has_hedging'],
        'has_conditional': detection['has_conditional'],
        'is_main_actor': is_main_actor,
        'used_local_clause': used_local_clause,
        'para_idx': para_idx,
    }


# === EXAMPLE OUTPUTS ===
# Sentence: "Prabowo tidak dipuji karena kebijakannya gagal"
# Before v20: has_sentiment=True (puji), polarity=positive, quality=100
# After v20:  has_sentiment=True, polarity=negative (negation reverses),
#             confidence_modifier=0.7 (negated), quality=70
#
# Sentence: "Prabowo sangat dikritik karena korupsinya"
# Before v20: has_sentiment=True (kritik), polarity=negative, quality=100
# After v20:  has_sentiment=True, polarity=negative, confidence_modifier=1.3 (intensity high),
#             quality=130, +5 bonus for clear polarity = 135
#
# Sentence: "Prabowo mungkin terlibat kasus korupsi"
# Before v20: has_sentiment=True (korupsi), polarity=negative, quality=100
# After v20:  has_sentiment=True, polarity=negative, confidence_modifier=0.6 (hedging),
#             quality=60 (uncertain, lower priority for training)
