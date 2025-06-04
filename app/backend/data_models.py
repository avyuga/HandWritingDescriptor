from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    request_id: str
    prediction: str | None
    confidence: float | None
    status: str

class UpdateRequest(BaseModel):
    request_id: str
    rating: int | None = Field(..., ge=1, le=5, description="Rating from 1 to 5")

class TranscribationRequest(BaseModel):
    request_id: str
    transcription: str | None = Field(..., min_length=1, description="User's transcription of the text")

class SimpleResponse(BaseModel):
    success: bool
    message: str
