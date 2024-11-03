from sentence_transformers import CrossEncoder
from fastapi import HTTPException
import torch
from typing import Union

# import payload classes
from ..payload.request.list_rerank_request_dto import ListRerankRequestDto
from ..payload.request.list_single_rerank_request_dto import ListSingleRerankRequestDto
from ..payload.response.list_rerank_response_dto import (
    ListRerankResponseDto,
    RerankResult,
)
from ..payload.response.list_single_rerank_response_dto import (
    ListSingleRerankResponseDto,
    RerankSingleResult,
)


def rerank_sentences_pair(request: ListRerankRequestDto):
    # Load the specified cross-encoder model
    try:
        if request.model in ["jinaai/jina-reranker-v1-turbo-en","jinaai/jina-reranker-v1-turbo-en"]:
            model = CrossEncoder(
                request.model,
                trust_remote_code=True,
            )
        else:
            model = CrossEncoder(request.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model loading error: {str(e)}")

    # Prepare the sentence pairs for prediction
    sentence_pairs = [(pair.text1, pair.text2) for pair in request.pairs]

    # Predict similarity scores
    scores = model.predict(sentence_pairs)

    # Build the response object with scores and original pairs
    response = ListRerankResponseDto(
        result=[
            RerankResult(
                text1=pair.text1,
                text2=pair.text2,
                score=score,
                normalized_score=sigmoid(score),
            )
            for pair, score in zip(request.pairs, scores)
        ]
    )
    return response


# The function to rerank the sentences
def rerank_sentences_single(request: ListSingleRerankRequestDto):
    # Load the specified cross-encoder model
    try:
        if request.model in ["jinaai/jina-reranker-v2-base-multilingual"]:
            model = CrossEncoder(
                request.model,
                automodel_args={"torch_dtype": "auto"},
                trust_remote_code=True,
            )
        else:
            model = CrossEncoder(request.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model loading error: {str(e)}")

    # Prepare the sentence pairs for prediction with main_text
    sentence_pairs = [
        (request.main_text, entity.text) for entity in request.entity_list
    ]

    # Predict similarity scores
    scores = model.predict(sentence_pairs)

    # Combine scores with the original entity data
    results = [
        RerankSingleResult(
            text=entity.text,
            key_id=entity.key_id,
            score=score,
            normalized_score=sigmoid(score),  # Apply sigmoid for normalization
        )
        for entity, score in zip(request.entity_list, scores)
    ]

    # Build and return the response
    response = ListSingleRerankResponseDto(result=results)
    return response


def sort_by_score(
    response: Union[ListRerankResponseDto, ListSingleRerankResponseDto],
    reverse: bool = False,
    key: str = "score",
):
    sorted_result = sorted(
        response.result, key=lambda x: getattr(x, key), reverse=reverse
    )

    # Return the correct response type based on the input type
    if isinstance(response, ListRerankResponseDto):
        return ListRerankResponseDto(result=sorted_result)
    elif isinstance(response, ListSingleRerankResponseDto):
        return ListSingleRerankResponseDto(result=sorted_result)
    else:
        raise ValueError("Unsupported response type")


def sigmoid(x):
    x_tensor = torch.tensor(x)  # Convert x to a Tensor
    return 1 / (1 + torch.exp(-x_tensor))
