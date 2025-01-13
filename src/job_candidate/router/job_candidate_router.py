from fastapi import APIRouter
from ..service import  job_candidate_service

# Request payload
from ..payload.request.job_to_candidates_matching_request_dto import (
  JobToCandidatesMatchingRequestDTO,
  CandidateToJobsMatchingRequestDTO
)

router = APIRouter(
    prefix="/api/job-candidate",
)

@router.post("/job-match-candidates")
async def job_match_candidates(job_to_candidates_matching_request_Dto: JobToCandidatesMatchingRequestDTO):
    results = job_candidate_service.get_matching(
        job_to_candidates_matching_request_Dto=job_to_candidates_matching_request_Dto, type="candidates")
    return results

@router.post("/candidate-match-jobs")
async def candidate_match_jobs(candidate_to_jobs_matching_request_Dto: CandidateToJobsMatchingRequestDTO):
    results = job_candidate_service.get_matching(
        candidate_to_jobs_matching_request_Dto=candidate_to_jobs_matching_request_Dto, type="jobs")
    return results