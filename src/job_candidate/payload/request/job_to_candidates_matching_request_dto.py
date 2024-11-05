from typing import List, Set, Optional
from pydantic import BaseModel

class JobParsedDataDTO(BaseModel):
    jobTitle: Optional[str]
    yearsOfExperience: Optional[int]
    yearsOfExperienceSkills: Optional[List[str]] = []
    qualifications: Optional[List[str]] = []
    skills: Optional[List[str]] = []
    experienceRole: Optional[List[str]] = []
    salary: Optional[str]
    keyWords: Optional[List[str]] = []
    location: Optional[str]

class CandidateParsedDataDTO(BaseModel):
    candidateId: Optional[int]
    candidateQualifications: Optional[List[str]] = List()
    candidateLanguages: Optional[List[str]] = List()
    candidateSkills: Optional[List[str]] = List()
    candidateJobTitles: Optional[List[str]] = List()
    candidateNationality: Optional[str]
    candidateDetails: Optional[str]
    candidateFieldOfStudy: Optional[List[str]] = List()
    candidateWorkExperiences: Optional[str]

class JobToCandidatesMatchingRequestDTO(BaseModel):
    candidatesData: List[CandidateParsedDataDTO]
    jobData: JobParsedDataDTO