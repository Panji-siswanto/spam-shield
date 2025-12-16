from fastapi import FastAPI
from pydantic import BaseModel

from model.nb_agent import NBAgent
from model.bert_agent import BertAgent
from model.hybrid_agent import HybridAgent

# ----------------------
# App
# ----------------------
app = FastAPI(title="Spam Shield API")

# ----------------------
# Load models once
# ----------------------
nb_agent = NBAgent()
bert_agent = BertAgent()
hybrid_agent = HybridAgent()


# ----------------------
# Request schema
# ----------------------
class TextRequest(BaseModel):
    text: str


# ----------------------
# Routes
# ----------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Spam Shield API is running"}


@app.post("/predict/nb")
def predict_nb(req: TextRequest):
    return nb_agent.smart_predict(req.text)


@app.post("/predict/bert")
def predict_bert(req: TextRequest):
    return bert_agent.predict(req.text)


@app.post("/predict/hybrid")
def predict_hybrid(req: TextRequest):
    return hybrid_agent.predict(req.text)
