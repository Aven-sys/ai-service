from ..util import embedding_util
from ..payload.request.embedding_request_dto import EmbeddingRequestDto

def get_embedding(embedding_request_dto: EmbeddingRequestDto):
    text = embedding_request_dto.text
    model_name = embedding_request_dto.model_name
    normalize = embedding_request_dto.normalize
    return embedding_util.generate_embeddings(text, model_name, normalize)
