from fastapi import APIRouter
from ..service import  job_candidate_service
from common.util.langchain_pydantic_model_generator import print_pydantic_instance

# Request payload
from ..payload.request.job_to_candidates_matching_request_dto import JobToCandidatesMatchingRequestDTO, JobToCandidatesMatchingRequestV2DTO
from common.util.langchain_pydantic_model_generator import print_pydantic_instance

router = APIRouter(
    prefix="/api/job-candidate",
)

@router.post("/job-match-candidates")
async def job_match_candidates(job_to_candidates_matching_request_Dto: JobToCandidatesMatchingRequestDTO):
    print_pydantic_instance(job_to_candidates_matching_request_Dto)
    job_candidate_service.get_job_to_candidates_matching(job_to_candidates_matching_request_Dto)
    return "Hello"

@router.post("/job-match-candidates/v3")
async def job_match_candidates(job_to_candidates_matching_request_Dto: JobToCandidatesMatchingRequestV2DTO):
    results = job_candidate_service.get_job_to_candidates_matching_v3(job_to_candidates_matching_request_Dto)
    return results
