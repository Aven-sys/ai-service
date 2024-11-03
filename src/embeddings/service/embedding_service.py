from ..util import embedding_util
from ..payload.request.embedding_request_dto import EmbeddingRequestDto
from ..payload.request.embedding_similarity_request_dto import EmbeddingSimilarityRequestDto
from ..util.similarity_matcher_util import SimilarityMatcher

def get_embedding(embedding_request_dto: EmbeddingRequestDto):
    text = embedding_request_dto.text
    model_name = embedding_request_dto.model
    normalize = embedding_request_dto.normalize
    return embedding_util.generate_embeddings(text, model_name, normalize)

def get_similarity(embedding_similarity_request_dto: EmbeddingSimilarityRequestDto):
    similarity_matcher = SimilarityMatcher(embedding_similarity_request_dto)
    print("CLASS: ", similarity_matcher)
    results = similarity_matcher.match_score(embedding_similarity_request_dto)
    return results
