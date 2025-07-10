from datetime import date
from typing import Optional
from pydantic import BaseModel, HttpUrl

class TextRequest(BaseModel):
    text: str
    url: Optional[HttpUrl] = None
    date: Optional[str] = None

class CheckResponse(BaseModel):
    traffic_light: str
    confidence: float
    explanation: str
    source_url: Optional[str] = None
