from ..payload.request.list_rerank_request_dto import ListRerankRequestDto
from ..payload.response.list_rerank_response_dto import ListRerankResponseDto, RerankResult
from sentence_transformers import CrossEncoder
from fastapi import HTTPException
import torch


def rerank_sentences(request: ListRerankRequestDto):
    # Load the specified cross-encoder model
    try:
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
            RerankResult(text1=pair.text1, text2=pair.text2, score=score, normalized_score=sigmoid(score))
            for pair, score in zip(request.pairs, scores)
        ]
    )

    return response

def sort_by_score(response: ListRerankResponseDto, reverse: bool = False, key: str = "score"):
    # return in ListRerankResponseDto 
    return ListRerankResponseDto(result=sorted(response.result, key=lambda x: getattr(x, key), reverse=reverse))

def sigmoid(x):
    x_tensor = torch.tensor(x)  # Convert x to a Tensor
    return 1 / (1 + torch.exp(-x_tensor))