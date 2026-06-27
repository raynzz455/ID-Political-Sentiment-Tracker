-- ============================================================
-- HOTFIX: batch_insert_raw_texts sets ingested_month explicitly
-- ============================================================
-- ROOT CAUSE (confirmed in production):
--   PostgreSQL partition routing happens BEFORE the BEFORE INSERT
--   trigger runs. When ingested_month is NULL at INSERT time, PG
--   cannot find a matching partition and throws 23514 "no partition
--   found" with "(ingested_month) = (null)". The trigger never gets
--   a chance to fill it, even when the trigger function is correct.
--
-- FIX: Drop the unreliable trigger. Compute ingested_month + ingested_at
--   inside the RPC and pass them explicitly in the INSERT.
--
-- Also: GRANT EXECUTE to service_role (required for SECURITY DEFINER
--   functions to be callable by the Edge Function / NLP worker).
--
-- Idempotent. Safe to re-run.
-- ============================================================

-- Step 1: Drop trigger yang tidak bekerja
DROP TRIGGER IF EXISTS set_raw_texts_month ON raw_texts;
DROP TRIGGER IF EXISTS set_sentiment_scores_month ON sentiment_scores;
DROP FUNCTION IF EXISTS trg_set_partition_month();

-- Step 2: Recreate batch_insert dengan ingested_month eksplisit
CREATE OR REPLACE FUNCTION batch_insert_raw_texts(p_items JSONB)
RETURNS TABLE(inserted_count INTEGER, duplicate_count INTEGER)
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  v_item JSONB;
  v_hash TEXT;
  v_new  TEXT;
  ins    INT := 0;
  dup    INT := 0;
  v_now  TIMESTAMPTZ := NOW();
  v_month DATE := date_trunc('month', NOW())::date;
BEGIN
  FOR v_item IN SELECT * FROM jsonb_array_elements(p_items) LOOP
    v_hash := encode(digest((v_item->>'text')::bytea, 'sha256'), 'hex');

    INSERT INTO raw_text_hashes (text_hash) VALUES (v_hash)
    ON CONFLICT (text_hash) DO NOTHING
    RETURNING text_hash INTO v_new;

    IF v_new IS NOT NULL THEN
      INSERT INTO raw_texts (
        source, source_id, title, source_url, image_url,
        text, text_hash, metadata, published_at,
        ingested_at, ingested_month
      ) VALUES (
        v_item->>'source',
        v_item->>'source_id',
        NULLIF(v_item->>'title', ''),
        NULLIF(v_item->>'source_url', ''),
        NULLIF(v_item->>'image_url', ''),
        v_item->>'text',
        v_hash,
        COALESCE(v_item->'metadata', '{}'),
        NULLIF(v_item->>'published_at', '')::timestamptz,
        v_now,
        v_month
      );
      ins := ins + 1;
    ELSE
      dup := dup + 1;
    END IF;
  END LOOP;

  RETURN QUERY VALUES (ins, dup);
END;
$$;

-- Step 3: Fix insert_sentiment_score dengan scored_month eksplisit
CREATE OR REPLACE FUNCTION insert_sentiment_score(
  p_raw_text_id  UUID,
  p_entity_id    UUID,
  p_label        TEXT,
  p_neg          REAL,
  p_neu          REAL,
  p_pos          REAL,
  p_confidence   REAL,
  p_aspect       TEXT DEFAULT NULL,
  p_model_version TEXT DEFAULT 'indobert-v1'
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE v_id UUID;
BEGIN
  INSERT INTO sentiment_scores (
    raw_text_id, entity_id, aspect,
    score_negative, score_neutral, score_positive,
    label, confidence, model_version,
    scored_at, scored_month
  ) VALUES (
    p_raw_text_id, p_entity_id, p_aspect,
    p_neg, p_neu, p_pos,
    p_label, p_confidence, p_model_version,
    NOW(), date_trunc('month', NOW())::date
  )
  RETURNING id INTO v_id;
  RETURN v_id;
END;
$$;

-- Step 4: Grant yang tadi missing
GRANT EXECUTE ON FUNCTION batch_insert_raw_texts(JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION insert_sentiment_score(UUID, UUID, TEXT, REAL, REAL, REAL, REAL, TEXT, TEXT) TO service_role;
