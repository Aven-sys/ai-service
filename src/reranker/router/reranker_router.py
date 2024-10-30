from fastapi import APIRouter
from ..service import reranker_service

# Request payload
from ..payload.request.list_rerank_request_dto import ListRerankRequestDto

router = APIRouter(
    prefix="/api/rerank",
)


@router.post("/")
async def single_rerank(list_rerank_request_dto: ListRerankRequestDto):
    rerank_result = reranker_service.list_rerank(list_rerank_request_dto)
    return rerank_result
