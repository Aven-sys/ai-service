from fastapi import APIRouter
from ..service import embedding_service

# Request payload
from ..payload.request.embedding_request_dto import EmbeddingRequestDto
from ..payload.request.embedding_similarity_request_dto import EmbeddingSimilarityRequestDto

router = APIRouter(
    prefix="/api/embeddings",
)

@router.post("")
async def embedding_test(
   embedding_request_dto: EmbeddingRequestDto 
):
    embedding = embedding_service.get_embedding(embedding_request_dto)
    return embedding

@router.post("/similarity-test")
async def similarity_test(
   embedding_similarity_request_dto: EmbeddingSimilarityRequestDto 
):
    similarity = embedding_service.get_similarity(embedding_similarity_request_dto)
    return similarity
