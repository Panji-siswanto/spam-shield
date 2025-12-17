from fastapi import FastAPI

from api.nb_routes import router as nb_router
from api.bert_routes import router as bert_router
from api.hybrid_routes import router as hybrid_router

app = FastAPI(title="Spam Shield API")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Spam Shield API is running"}


app.include_router(nb_router)
app.include_router(bert_router)
app.include_router(hybrid_router)
