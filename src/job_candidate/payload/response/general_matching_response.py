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
    
class MatchingCriteriaDTO(BaseModel):
    salary: Optional[GeneralMatchingResponse] = None
    years_of_experience: Optional[GeneralMatchingResponse] = None
    qualifications: Optional[GeneralMatchingResponse] = None
    location: Optional[GeneralMatchingResponse] = None
    skills: Optional[GeneralMatchingResponse] = None
    job_title: Optional[GeneralMatchingResponse] = None
    experience: Optional[GeneralMatchingResponse] = None
    keywords: Optional[GeneralMatchingResponse] = None
    total_score: Optional[float] = None

class MatchingListDTO(BaseModel):
    results: Optional[List[MatchingCriteriaDTO]] = None