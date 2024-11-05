from fastapi import APIRouter
from ..service import preprocess_service
from common.util.langchain_pydantic_model_generator import print_pydantic_instance

# Request payload
from ..payload.request.preprocess_text_request_dto import PreprocessTextRequestDto

router = APIRouter(
    prefix="/api/job-candidate",
)

@router.post("/job-match-candidates")
async def job_match_candidates(preprocess_text_request_dto: PreprocessTextRequestDto):
    preprocess_result = preprocess_service.preprocess_text_single(preprocess_text_request_dto)
    return preprocess_result

