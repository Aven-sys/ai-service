from pydantic import BaseModel
from typing import Optional

class EmbeddingSimilarityRequestDto(BaseModel):
    text1: str
    text2: str
    type: str
    text_match_type: Optional[str] = "exact"
    model: str
    normalize: Optional[bool] = True
    top_k: Optional[int] = 5
    threshold: Optional[float] = 0.0


    