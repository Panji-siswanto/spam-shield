from pydantic import BaseModel
from typing import List


class TextRequest(BaseModel):
    text: str


class BatchTextRequest(BaseModel):
    texts: List[str]


class ConversationRequest(BaseModel):
    messages: List[str]
