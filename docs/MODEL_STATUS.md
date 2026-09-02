# Model Status Report — ID-Political-Sentiment-Tracker

> Last updated: August 2026 | Branch: feature/finetuning-optimized | PR: #1

## 1. Arsitektur Model Saat Ini

### Base Models (HuggingFace)
| Role | Model ID | Format |
|------|----------|--------|
| Relevancy Gate | apriandito/indobert-relevancy-classifier | Sentence-pair NLI |
| Sentiment Classifier | apriandito/indobert-sentiment-classifier | Sentence-pair NLI |
| Fallback | taufiqdp/indonesian-sentiment | Single text |

### Pipeline (2-Stage Gated)
```
Article -> Entity Resolution -> Context Extraction
                                    |
                        Relevancy Gate (is context about entity?)
                                    | relevant
                        Sentiment Classifier (pos/neu/neg toward entity)
                                    |
                        Confidence Check (tau=0.70)
                           |                |
                     confident           deferred -> human/LLM
```

## 2. Dataset Status
- Total: 909 rows, all labeled, 0 unverified
- Distribution: neutral 66%, positive 18%, negative 16%
- High confidence (>=0.85): 38% | Mid (0.6-0.84): 46% | Low (<0.6): 16%

## 3. Finetuning Method — M5 (Anti-Overconfidence)

### Hyperparameters
- LoRA r=32, alpha=64
- Focal gamma=2.5 + class weights 1/sqrt(freq)
- Label smoothing=0.05 (caps max confidence ~0.90)
- Temperature=1.3 (softens softmax)
- Confidence tau=0.70
- SWA enabled (epoch 7+)
- 10 epochs, early stop patience=3

### Anti-Overconfidence Test Results
| Method | ECE | Conf>0.9 | Status |
|--------|-----|----------|--------|
| M1 baseline | 0.130 | 84% | Overconfident |
| M4 focal+weights | 0.092 | 85% | Still overconfident |
| M5 focal+smoothing_0.05+temp_1.3 | 0.149 | 23% | BALANCED (recommended) |
| M6 focal+smoothing_0.1+temp_1.5 | 0.203 | 2% | Too conservative |

## 4. Steps to Upload to HuggingFace

### Prerequisites
1. HF account: https://huggingface.co/join
2. Token: https://huggingface.co/settings/tokens (write scope)
3. `pip install huggingface_hub`

### Upload Process
```bash
huggingface-cli login  # paste token
python finetuning/scripts/finetune.py --task sentiment
python finetuning/scripts/evaluate.py --task sentiment
export HF_TOKEN=hf_your_token
python finetuning/scripts/upload_huggingface.py --task sentiment
python finetuning/scripts/upload_huggingface.py --task relevancy
```

### Model URLs (after upload)
- huggingface.co/raynzz455/id-political-sentiment-sentiment-v1
- huggingface.co/raynzz455/id-political-sentiment-relevancy-v1

### Update Production
```python
# In packages/nlp/sentiment_model.py:
SENTIMENT_MODEL_ID = "raynzz455/id-political-sentiment-sentiment-v1"
RELEVANCY_MODEL_ID = "raynzz455/id-political-sentiment-relevancy-v1"
```

## 5. Target Metrics
| Metric | Target | Status |
|--------|--------|--------|
| macro-F1 | >=0.90 | Pending GPU run |
| ECE | <=0.15 | Simulation: 0.149 |
| Kept-set acc (tau=0.70) | >=0.97 | Simulation: 100% |
| Coverage at 97% | >=35% | Simulation: 37% |

## 6. Limitations
- 62% labels are heuristic (not LLM-verified)
- Context still v17 (patch v18.1 not deployed)
- Entity resolution v14.2 (v15 in development)
- GPU testing pending

## 7. Next Steps
1. Merge PR #1
2. Deploy entity v15 + context v18.1
3. Re-run pipeline on 200 articles
4. Re-label via LLM
5. Finetune on Colab GPU (M5 method)
6. Verify metrics
7. Upload to HuggingFace
8. Update production model
