from fastapi import APIRouter
from ..service import embedding_service

# Request payload
from ..payload.request.embedding_request_dto import EmbeddingRequestDto

router = APIRouter(
    prefix="/api/embeddings",
)


@router.post("/")
async def embedding_test(
   embedding_request_dto: EmbeddingRequestDto 
):
    print("Request: ", embedding_request_dto)
    embedding = embedding_service.get_embedding()
    print("Embedding: ", embedding) 
    return embedding
