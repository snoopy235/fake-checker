from fastapi import FastAPI
from app.schemas import TextRequest, CheckResponse
from app.logic import check_text

app = FastAPI()

@app.post("/check_fake", response_model=CheckResponse)
def check_fake(request: TextRequest):
    return check_text(
        text=request.text,
        url=request.url,
        date=request.date
    )
