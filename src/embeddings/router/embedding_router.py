from fastapi import APIRouter
from ..service import embedding_service

# Request payload
from ..payload.request.embedding_request_dto import EmbeddingRequestDto
from ..payload.request.embedding_similarity_request_dto import EmbeddingSimilarityRequestDto
from common.util.langchain_pydantic_model_generator import print_pydantic_instance

router = APIRouter(
    prefix="/api/embeddings",
)

@router.post("")
async def embedding_test(
   embedding_request_dto: EmbeddingRequestDto 
):
    print("Request: ", embedding_request_dto)
    embedding = embedding_service.get_embedding(embedding_request_dto)
    print("Embedding: ", embedding) 
    return embedding

@router.post("/similarity-test")
async def similarity_test(
   embedding_similarity_request_dto: EmbeddingSimilarityRequestDto 
):
    similarity = embedding_service.get_similarity(embedding_similarity_request_dto)
    # print("Similarity: ", similarity) 
    return similarity
