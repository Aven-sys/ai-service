import nltk
import math
from typing import Optional, List
from pydantic import BaseModel
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from ..payload.request.embedding_similarity_request_dto import (
    EmbeddingSimilarityRequestDto,
)
from ..payload.response.embedding_similarity_response_dto import (
    EmbeddingSimilarityResponseDto,
    MatchResult,
)

# Initialize the NLTK tokenizer
# nltk.download('punkt')


class SimilarityMatcher:
    def __init__(self, request: EmbeddingSimilarityRequestDto):
        self.model_name = request.model
        self.threshold = request.threshold
        self.top_k = request.top_k
        self.normalize = request.normalize
        self.method = request.type
        self.match_type = request.text_match_type
        self.embedding_model = (
            SentenceTransformer(self.model_name)
            if request.type == "embedding"
            else None
        )
        self.tfidf_vectorizer = TfidfVectorizer() if request.type == "tfidf" else None

    def match_score(
        self, request: EmbeddingSimilarityRequestDto
    ) -> EmbeddingSimilarityResponseDto:
        # Perform matching based on the selected match type
        if request.text_match_type == "exact":
            matches = self._match_exact(request.text1, request.text2, request.type)
        elif request.text_match_type == "sentence":
            matches = self._match_sentences(request.text1, request.text2, request.type)
        elif request.text_match_type == "word":
            matches = self._match_words(request.text1, request.text2, request.type)
        else:
            raise ValueError(
                "Invalid text_match_type. Choose from 'exact', 'sentence', or 'word'."
            )

        # Return structured response
        return EmbeddingSimilarityResponseDto(
            matches=matches,
            method=request.type,
            match_type=request.text_match_type,
            top_k=request.top_k,
            threshold=request.threshold,
        )

    def _match_exact(self, text1, text2, method) -> List[MatchResult]:
        score = self._calculate_score(text1, text2, method)
        return (
            [MatchResult(text1_segment=text1, text2_segment=text2, score=score)]
            if score >= self.threshold
            else []
        )

    def _match_sentences(self, text1, text2, method) -> List[MatchResult]:
        # Split texts into sentences
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        return self._calculate_matrix(sentences1, sentences2, method)

    def _match_words(self, text1, text2, method) -> List[MatchResult]:
        # Tokenize texts into words
        words1 = word_tokenize(text1)
        words2 = word_tokenize(text2)
        return self._calculate_matrix(words1, words2, method)

    def _calculate_matrix(self, list1, list2, method, alpha=0.1) -> List[MatchResult]:
        print("List1: ", list1)
        print("List2: ", list2)
        results = []

        if method == "bm25":
            # Tokenize list2 elements for BM25 corpus
            tokenized_list2 = [item.split() for item in list2]
            bm25 = BM25Okapi(tokenized_list2)

            for item1 in list1:
                item1_tokens = item1.split()
                scores = bm25.get_scores(item1_tokens)

                # Apply sigmoid normalization to each score
                normalized_scores = [sigmoid(score, alpha=alpha) for score in scores]
                print("Threshold: ", self.threshold)
                for j, score in enumerate(normalized_scores):
                    print(f"Sigmoid Normalized BM25 Score between '{item1}' and '{list2[j]}': {score:.4f}")

                    if score >= self.threshold:
                        results.append(
                            MatchResult(
                                text1_segment=item1,
                                text2_segment=list2[j],
                                score=score
                            )
                        )
        else:
            for item1 in list1:
                for item2 in list2:
                    score = self._calculate_score(item1, item2, method)
                    if score >= self.threshold:
                        results.append(
                            MatchResult(
                                text1_segment=item1,
                                text2_segment=item2,
                                score=score
                            )
                        )

        results = sorted(results, key=lambda x: x.score, reverse=True)[:self.top_k]
        return results


    def _calculate_score(self, text1, text2, method):
        if method == "embedding":
            return self._embedding_score(text1, text2)
        elif method == "tfidf":
            return self._tfidf_score(text1, text2)
        elif method == "bm25":
            # This method is used only for word or sentence-level BM25 comparisons
            raise ValueError(
                "BM25 should be calculated in _calculate_matrix for dynamic handling."
            )
        else:
            raise ValueError(
                "Invalid type specified. Choose from 'embedding', 'tfidf', or 'bm25'."
            )

    def _embedding_score(self, text1, text2):
        query_embedding = self.embedding_model.encode(
            text1, normalize_embeddings=self.normalize
        )
        doc_embedding = self.embedding_model.encode(
            text2, normalize_embeddings=self.normalize
        )
        score = util.cos_sim(query_embedding, doc_embedding).item()
        return score

    def _tfidf_score(self, text1, text2):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        score = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray().item()
        return score


def sigmoid(x, alpha=1.0):
    return 1 / (1 + math.exp(-alpha * x))