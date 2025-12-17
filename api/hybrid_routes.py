from fastapi import APIRouter
from model.hybrid_agent import HybridAgent
from api.schemas import TextRequest, BatchTextRequest

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
