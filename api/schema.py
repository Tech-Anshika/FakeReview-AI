from pydantic import BaseModel

class ReviewInput(BaseModel):
    text: str
    rating: int | None = None
    verified_purchase: bool | None = None
    helpful_vote: int | None = None
    user_review_burst: int | None = None

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    text_risk: float
    behavior_score: float
