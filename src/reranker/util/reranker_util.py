from sentence_transformers import CrossEncoder
from fastapi import HTTPException
import torch
from typing import Union
import nltk
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics.pairwise import cosine_similarity

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
    
def preprocess_text(text):
    # Lowercase, remove special characters, and split by whitespace
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).split()

def preprocess_text_string(text):
    # Lowercase, remove special characters, and split by whitespace
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

# Split and preprocess sentences
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return [preprocess_text(sentence) for sentence in sentences]  # Assuming preprocess_text if required

def remove_stopwords_string(text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)

# BM25 Scoring Function
def compare_documents_with_sentence_bm25(request: ListSingleRerankRequestDto, min_score=0, max_score=10) -> ListSingleRerankResponseDto:
    """
    Compare a main document with multiple entities using BM25, scoring each entity based on its alignment with the main document.
    """
    # Split main_text into sentences for BM25 corpus
    main_sentences = split_into_sentences(request.main_text)
    # bm25 = BM25Okapi(main_sentences, b=0.75, k1=1.2)
    print("main_sentences: ", main_sentences)
    # Prepare results
    results = []

    # Process each entity in the request
    for entity in request.entity_list:
        # Split entity text into sentences
        entity_sentences = split_into_sentences(entity.text)
        print("entity_sentences: ", entity_sentences)
        bm25 = BM25Okapi(entity_sentences, b=0.75, k1=1.2)

        # Score each sentence in entity against the main_text corpus
        scores = []
        # for query_sentence in entity_sentences:
        for query_sentence in main_sentences:
            # Get BM25 scores for the query sentence against the main_text corpus
            sentence_scores = bm25.get_scores(query_sentence)
            # if sentence_scores:  # Ensure there are scores to take the max from
            scores.append(max(sentence_scores))  # Take the highest score for the query sentence

        # Calculate average score and normalized score
        average_score = sum(scores) / len(scores) if scores else 0
        normalized_scores = [
            (score - min_score) / (max_score - min_score) if max_score > min_score else 0
            for score in scores
        ]
        normalized_average_score = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0

        # Append result for the entity
        results.append(RerankSingleResult(
            text=entity.text,
            key_id=entity.key_id,
            score=average_score,
            normalized_score=normalized_average_score
        ))

    # Return the response DTO
    return ListSingleRerankResponseDto(result=results)

def rerank_sentences_single_tfidf(request: ListSingleRerankRequestDto) -> ListSingleRerankResponseDto:
    # Load the TF-IDF model
    try:
        model = TfidfVectorizer(stop_words='english')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model loading error: {str(e)}")

    # Fit TF-IDF on main_text only
    main_vector = model.fit_transform([remove_stopwords_string(preprocess_text_string(request.main_text))])

    results = []
    for entity in request.entity_list:
        # Transform entity text using the TF-IDF model trained on main_text
        entity_vector = model.transform([remove_stopwords_string(preprocess_text_string(entity.text))])

        # Compute cosine similarity between main_text and entity text
        score = cosine_similarity(main_vector, entity_vector).flatten()[0]

        # Append the result with score and normalized score
        results.append(
            RerankSingleResult(
                text=entity.text,
                key_id=entity.key_id,
                score=score,
                normalized_score=sigmoid(score)  # Apply sigmoid for normalization
            )
        )

    # Build and return the response
    response = ListSingleRerankResponseDto(result=results)
    return response


def sigmoid(x):
    x_tensor = torch.tensor(x)  # Convert x to a Tensor
    return 1 / (1 + torch.exp(-x_tensor))


