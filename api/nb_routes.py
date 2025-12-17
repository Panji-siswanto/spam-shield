from fastapi import APIRouter
from model.nb_agent import NBAgent
from api.schemas import TextRequest, BatchTextRequest

router = APIRouter(prefix="/predict/nb", tags=["Naive Bayes"])

nb_agent = NBAgent()


@router.post("/")
def predict_nb(req: TextRequest):
    return nb_agent.smart_predict(req.text)


@router.post("/batch")
def predict_nb_batch(req: BatchTextRequest):
    results = []
    for text in req.texts:
        results.append({"text": text, "result": nb_agent.smart_predict(text)})
    return results
