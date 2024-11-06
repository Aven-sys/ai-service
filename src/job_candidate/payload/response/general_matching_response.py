from typing import List, Any, Optional
from pydantic import BaseModel

class MatchResult(BaseModel):
    source_item: Optional[str] = None
    target_item: Optional[str] = None
    similarity_score: Optional[float] = None

class GeneralMatchingResponse(BaseModel):
    matches: Optional[List[MatchResult]] = None
    overall_score: Optional[float] = None

    @classmethod
    def empty_response(cls):
        # Generate an instance with all fields set to None
        return cls(matches=None, overall_score=None)

    @classmethod
    def empty_response(cls):
        # Generate an instance with all None fields
        return cls(matches=None, overall_score=None)