from pydantic import BaseModel, Field
from typing import List, Optional, Any


class ResearchRequest(BaseModel):
    query: str
    max_web_results: int = 5
    max_rag_chunks: int = 5

class ResearchResponse(BaseModel):
    summary: str
    sources: List[str] = []
    web_results: List[dict] = []
    rag_passages: List[dict] = []


