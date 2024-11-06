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
    candidateQualifications: Optional[List[str]] = []
    candidateLanguages: Optional[List[str]] = []
    candidateSkills: Optional[List[str]] = []
    candidateJobTitles: Optional[List[str]] = []
    candidateNationality: Optional[str]
    candidateDetails: Optional[str]
    candidateFieldOfStudy: Optional[List[str]] = []
    candidateWorkExperiences: Optional[str]
    candidateEducation: Optional[str]
    candidateLocation: Optional[str]
    candidateYearsOfExperience: Optional[int]

class JobToCandidatesMatchingRequestDTO(BaseModel):
    candidatesData: List[CandidateParsedDataDTO]
    jobData: JobParsedDataDTO