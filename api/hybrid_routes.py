from fastapi import APIRouter, Request
from model.hybrid_agent import HybridAgent
from api.schemas import TextRequest, BatchTextRequest
from api.schemas import ConversationRequest
from utils.context import build_context
import json
import logging


logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/predict/hybrid", tags=["Hybrid"])

hybrid_agent = HybridAgent()


@router.post("/")
async def predict_hybrid(req: TextRequest, request: Request):
    raw_body = await request.body()
    logger.error("RAW BODY RECEIVED: %s", raw_body)

    logger.error("PARSED TEXT: %s", req.text)

    return hybrid_agent.predict(req.text)


@router.post("/batch")
def predict_hybrid_batch(req: BatchTextRequest):
    results = []
    for text in req.texts:
        results.append({"text": text, "result": hybrid_agent.predict(text)})
    return results


@router.post("/context")
def predict_bert_context(req: ConversationRequest):
    context_text = build_context(req.messages)
    return {
        "context": context_text,
        "result": hybrid_agent.predict(context_text),
    }
