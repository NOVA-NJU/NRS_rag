from pydantic import BaseModel
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    text: str
    url: str
    title: str
    score: Optional[float] = None

class AnswerResponse(BaseModel):
    code: str = "200"
    answer: str
    sources: List[SourceDocument]

class ErrorResponse(BaseModel):
    error: str
    code: str = "404"
    details: Optional[str] = None