from typing import List, Optional
from pydantic import BaseModel

class MatchResult(BaseModel):
    text1_segment: str
    text2_segment: str
    score: float

class EmbeddingSimilarityResponseDto(BaseModel):
    matches: List[MatchResult]  # List of matching segments with scores
    method: str  # The similarity method used, e.g., "embedding", "tfidf", "bm25"
    match_type: str  # The type of match performed, e.g., "word", "sentence", "exact"
    top_k: Optional[int] = None  # Number of top matches returned, if applicable
    threshold: Optional[float] = None  # Score threshold applied