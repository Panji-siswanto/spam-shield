from fastapi import APIRouter
from model.nb_agent import NBAgent
from api.schemas import TextRequest, BatchTextRequest
from api.schemas import ConversationRequest
from utils.context import build_context

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


@router.post("/context")
def predict_bert_context(req: ConversationRequest):
    context_text = build_context(req.messages)
    return {
        "context": context_text,
        "result": nb_agent.predict(context_text),
    }
