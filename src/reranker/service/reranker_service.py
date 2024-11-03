from ..payload.request.list_rerank_request_dto import ListRerankRequestDto
from ..payload.request.list_single_rerank_request_dto import ListSingleRerankRequestDto
from ..util import reranker_util


def list_rerank(list_rerank_request_dto: ListRerankRequestDto):
    rereank_data = reranker_util.rerank_sentences_pair(list_rerank_request_dto)

    if list_rerank_request_dto.sorted == "ASC":
        return reranker_util.sort_by_score(rereank_data, reverse=False, key="score")
    elif list_rerank_request_dto.sorted == "DESC":
        return reranker_util.sort_by_score(rereank_data, reverse=True, key="score")

    return rereank_data


def list_single_rerank(list_single_rerank_request_dto: ListSingleRerankRequestDto):
    rereank_data = reranker_util.rerank_sentences_single(list_single_rerank_request_dto)

    if list_single_rerank_request_dto.sorted == "ASC":
        return reranker_util.sort_by_score(rereank_data, reverse=False, key="score")
    elif list_single_rerank_request_dto.sorted == "DESC":
        return reranker_util.sort_by_score(rereank_data, reverse=True, key="score")

    return rereank_data