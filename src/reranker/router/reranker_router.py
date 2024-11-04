from fastapi import APIRouter
from ..service import reranker_service
from common.util.langchain_pydantic_model_generator import print_pydantic_instance

# Request payload
from ..payload.request.list_rerank_request_dto import ListRerankRequestDto
from ..payload.request.list_single_rerank_request_dto import ListSingleRerankRequestDto

router = APIRouter(
    prefix="/api/rerank",
)

@router.post("/list-pair")
async def list_rerank(list_rerank_request_dto: ListRerankRequestDto):
    rerank_result = reranker_service.list_rerank(list_rerank_request_dto)
    return rerank_result

@router.post("/list-single")
async def list_rerank(list_single_rerank_request_dto: ListSingleRerankRequestDto):
    print_pydantic_instance(list_single_rerank_request_dto)
    rerank_result = reranker_service.list_single_rerank(list_single_rerank_request_dto)
    return rerank_result

@router.post("/list-single/bm25")
async def list_rerank(list_single_rerank_request_dto: ListSingleRerankRequestDto):
    rerank_result = reranker_service.list_single_rerank_bm25(list_single_rerank_request_dto)
    return rerank_result
