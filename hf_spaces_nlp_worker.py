"""
HuggingFace Spaces App — NLP Worker + MoE Runner
===================================================
Deploy this to HuggingFace Spaces (FREE, 16GB RAM) to run:
  - NLP Worker v16 (sentiment inference)
  - Entity Resolution MoE (6 experts)
  - Context Extraction MoE (5 experts)

HuggingFace Spaces Free Tier:
  - 2 vCPU, 16GB RAM (LEBIH BESAR dari GH Actions 7GB!)
  - 50GB ephemeral storage
  - Public: unlimited requests
  - Sleep after 48h idle (free tier)

DEPLOYMENT:
  1. Create new Space at huggingface.co/spaces
  2. Set SDK: Docker (for custom deps)
  3. Upload this file + Dockerfile + requirements.txt
  4. Set secrets: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  5. Space auto-deploys

USAGE (from GitHub Actions):
  POST /api/nlp/run   → trigger NLP worker
  POST /api/moe/run   → trigger MoE pipeline
  GET  /api/status    → check pipeline status
  GET  /api/health    → health check
"""
import os
import logging
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ID-Political-Sentiment NLP Worker", version="1.0")

# Lazy-loaded components (load on first request, not at startup)
_nlp_pipeline = None
_entity_moe = None
_context_moe = None


class NLPRunRequest(BaseModel):
    target: int = 100
    batch_size: int = 50
    run_all: bool = False


class MoERunRequest(BaseModel):
    limit: int = 50
    module: str = "entity"  # "entity" or "context"


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "loaded": {
            "nlp_pipeline": _nlp_pipeline is not None,
            "entity_moe": _entity_moe is not None,
            "context_moe": _context_moe is not None,
        }
    }


@app.post("/api/nlp/run")
async def run_nlp(request: NLPRunRequest, authorization: Optional[str] = Header(None)):
    """Trigger NLP worker to process queue.
    
    Runs in background (async) — returns immediately with job_id.
    """
    # Verify auth (optional, for security)
    expected_token = os.environ.get("HF_TOKEN", "")
    if expected_token and authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Lazy load NLP pipeline
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = _load_nlp_pipeline()
    
    # Run in background
    job_id = f"nlp_{asyncio.get_event_loop().time()}"
    asyncio.create_task(_run_nlp_background(request, job_id))
    
    return {
        "status": "started",
        "job_id": job_id,
        "target": request.target,
        "batch_size": request.batch_size
    }


async def _run_nlp_background(request: NLPRunRequest, job_id: str):
    """Run NLP worker in background."""
    try:
        logger.info(f"[{job_id}] Starting NLP worker...")
        
        # Import and run actual worker
        from packages.nlp.nlp_worker import main as nlp_main
        nlp_main(target=request.target, batch_size=request.batch_size, run_all=request.run_all)
        
        logger.info(f"[{job_id}] NLP worker completed")
    except Exception as e:
        logger.error(f"[{job_id}] NLP worker failed: {e}")


@app.post("/api/moe/run")
async def run_moe(request: MoERunRequest, authorization: Optional[str] = Header(None)):
    """Trigger MoE pipeline (entity + context extraction).
    
    Runs 6 entity experts + 5 context experts in parallel.
    Requires 4-6GB RAM — HuggingFace Spaces (16GB) can handle this.
    """
    expected_token = os.environ.get("HF_TOKEN", "")
    if expected_token and authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Lazy load MoE
    global _entity_moe, _context_moe
    if _entity_moe is None or _context_moe is None:
        _entity_moe, _context_moe = _load_moe()
    
    # Run in background
    job_id = f"moe_{asyncio.get_event_loop().time()}"
    asyncio.create_task(_run_moe_background(request, job_id))
    
    return {
        "status": "started",
        "job_id": job_id,
        "limit": request.limit,
        "module": request.module
    }


async def _run_moe_background(request: MoERunRequest, job_id: str):
    """Run MoE pipeline in background."""
    try:
        logger.info(f"[{job_id}] Starting MoE pipeline...")
        
        if request.module == "entity":
            await _run_entity_moe(request.limit)
        elif request.module == "context":
            await _run_context_moe(request.limit)
        else:
            # Run both
            await _run_entity_moe(request.limit)
            await _run_context_moe(request.limit)
        
        logger.info(f"[{job_id}] MoE pipeline completed")
    except Exception as e:
        logger.error(f"[{job_id}] MoE pipeline failed: {e}")


async def _run_entity_moe(limit: int):
    """Run entity resolution MoE on unprocessed articles."""
    from packages.shared.db_client import get_client
    from packages.entity.entity_resolution_moe import create_entity_moe_from_db
    import stanza
    
    sb = get_client()
    nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                         verbose=False, use_gpu=False, batch_size=32)
    
    moe = create_entity_moe_from_db(sb, stanza_nlp=nlp,
                                     enable_dbpedia=True,
                                     enable_embedding=True,
                                     enable_polyglot=True,
                                     enable_malaya=False)  # too large for free tier
    
    # Fetch unprocessed articles
    res = sb.table("raw_texts").select("id, title, text").eq("status", "validated") \
            .is_("entity_resolved_at", "null").limit(limit).execute()
    
    for art in (res.data or []):
        result = moe.resolve_to_db_format(art["text"], art["id"])
        # Insert to DB
        sb.table("article_entity_map").upsert(result["mappings"]).execute()
        sb.table("entity_mentions").upsert(result["mentions"]).execute()
        # Mark as resolved
        sb.table("raw_texts").update({"entity_resolved_at": "now()"}).eq("id", art["id"]).execute()


async def _run_context_moe(limit: int):
    """Run context extraction MoE on articles with entities."""
    from packages.shared.db_client import get_client
    from packages.context.context_extraction_moe import ContextExtractionMoE
    import stanza
    
    sb = get_client()
    nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                         verbose=False, use_gpu=False, batch_size=32)
    
    moe = ContextExtractionMoE(stanza_nlp=nlp, enable_embedding=True)
    
    # Fetch articles with entities but no context
    res = sb.table("raw_texts").select("id, title, text").eq("status", "validated") \
            .not_.is_("entity_resolved_at", "null") \
            .is_("context_extracted_at", "null").limit(limit).execute()
    
    for art in (res.data or []):
        # Get entity mentions for this article
        mentions_res = sb.table("entity_mentions").select("entity_id, start_offset, end_offset") \
            .eq("raw_text_id", art["id"]).execute()
        
        for m in (mentions_res.data or []):
            entity_name = sb.table("political_entities").select("canonical_name") \
                .eq("id", m["entity_id"]).execute().data[0]["canonical_name"]
            
            result = moe.extract_to_db_format(
                art["text"], entity_name, m["start_offset"],
                art["id"], m["entity_id"]
            )
            sb.table("entity_contexts").upsert(result).execute()
        
        # Mark as context-extracted
        sb.table("raw_texts").update({"context_extracted_at": "now()"}).eq("id", art["id"]).execute()


def _load_nlp_pipeline():
    """Lazy load NLP pipeline (3 models, ~1.8GB RAM)."""
    try:
        from packages.nlp.sentiment_model import get_pipeline
        pipeline = get_pipeline()
        logger.info("NLP pipeline loaded (3 models)")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load NLP pipeline: {e}")
        return None


def _load_moe():
    """Lazy load MoE components."""
    try:
        from packages.shared.db_client import get_client
        from packages.entity.entity_resolution_moe import create_entity_moe_from_db
        from packages.context.context_extraction_moe import ContextExtractionMoE
        import stanza
        
        sb = get_client()
        nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                             verbose=False, use_gpu=False, batch_size=32)
        
        entity_moe = create_entity_moe_from_db(sb, stanza_nlp=nlp,
                                                enable_dbpedia=True,
                                                enable_embedding=True,
                                                enable_polyglot=True,
                                                enable_malaya=False)
        
        context_moe = ContextExtractionMoE(stanza_nlp=nlp, enable_embedding=True)
        
        logger.info("MoE loaded (entity + context)")
        return entity_moe, context_moe
    except Exception as e:
        logger.error(f"Failed to load MoE: {e}")
        return None, None


@app.get("/api/status")
async def status():
    """Get pipeline status from Supabase."""
    from packages.shared.db_client import get_client
    sb = get_client()
    
    # Count articles by status
    statuses = {}
    for status in ["pending", "enriched", "validated", "queued", "processed"]:
        res = sb.table("raw_texts").select("id", count="exact").eq("status", status).execute()
        statuses[status] = res.count
    
    # Count contexts, entities, sentiments
    contexts = sb.table("entity_contexts").select("id", count="exact").execute().count
    entities = sb.table("article_entity_map").select("id", count="exact").execute().count
    sentiments = sb.table("sentiment_scores").select("id", count="exact").execute().count
    
    return {
        "articles": statuses,
        "entity_contexts": contexts,
        "entity_mappings": entities,
        "sentiment_scores": sentiments,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
