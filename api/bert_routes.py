from fastapi import APIRouter
from model.bert_agent import BertAgent
from api.schemas import TextRequest, BatchTextRequest

router = APIRouter(prefix="/predict/bert", tags=["BERT"])

bert_agent = BertAgent()


@router.post("/")
def predict_bert(req: TextRequest):
    return bert_agent.predict(req.text)


@router.post("/batch")
def predict_bert_batch(req: BatchTextRequest):
    results = []
    for text in req.texts:
        results.append({"text": text, "result": bert_agent.predict(text)})
    return results
