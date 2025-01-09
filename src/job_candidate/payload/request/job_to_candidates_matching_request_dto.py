from typing import List, Optional
from pydantic import BaseModel

class ParsedDataDTO(BaseModel):
    jobTitle: Optional[List[str]] = []
    yearsOfExperience: Optional[List[str]] = []
    qualifications: Optional[List[str]] = []
    skills: Optional[List[str]] = []
    experienceRole: Optional[List[str]] = []
    salary: Optional[List[str]] = []
    keyWords: Optional[List[str]] = []
    location: Optional[List[str]] = []

class JobToCandidatesMatchingRequestDTO(BaseModel):
    candidatesData: List[ParsedDataDTO]
    jobData: ParsedDataDTO

class CandidateToJobsMatchingRequestDTO(BaseModel):
    jobsData: List[ParsedDataDTO]
    candidateData: ParsedDataDTO