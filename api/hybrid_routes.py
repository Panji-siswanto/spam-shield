from fastapi import APIRouter
from model.hybrid_agent import HybridAgent
from api.schemas import TextRequest, BatchTextRequest
from api.schemas import ConversationRequest
from utils.context import build_context

router = APIRouter(prefix="/predict/hybrid", tags=["Hybrid"])

hybrid_agent = HybridAgent()


@router.post("/")
def predict_hybrid(req: TextRequest):
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
