import nltk
from typing import Optional, List
from pydantic import BaseModel
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from ..payload.request.embedding_similarity_request_dto import EmbeddingSimilarityRequestDto
from ..payload.response.embedding_similarity_response_dto import EmbeddingSimilarityResponseDto, MatchResult

# Initialize the NLTK tokenizer
# nltk.download('punkt')
# nltk.download('punkt_tab')

class SimilarityMatcher:
    def __init__(self, request: EmbeddingSimilarityRequestDto):
        self.model_name = request.model
        self.threshold = request.threshold
        self.top_k = request.top_k
        self.normalize = request.normalize
        self.method = request.type
        self.match_type = request.text_match_type
        self.embedding_model = SentenceTransformer(self.model_name) if request.type == "embedding" else None
        self.tfidf_vectorizer = TfidfVectorizer() if request.type == "tfidf" else None

    def match_score(self, request: EmbeddingSimilarityRequestDto) -> EmbeddingSimilarityResponseDto:
        if request.text_match_type == "word":
            matches = self._match_words(request.text1, request.text2, request.type)
        elif request.text_match_type == "sentence":
            matches = self._match_sentences(request.text1, request.text2, request.type)
        elif request.text_match_type == "exact":
            matches = self._match_full_text(request.text1, request.text2, request.type)
        else:
            raise ValueError("Invalid text_match_type. Choose from 'word', 'sentence', or 'exact'.")

        # Return structured response
        return EmbeddingSimilarityResponseDto(
            matches=matches,
            method=request.type,
            match_type=request.text_match_type,
            top_k=request.top_k,
            threshold=request.threshold
        )

    def _match_words(self, text1, text2, method) -> List[MatchResult]:
        words1 = word_tokenize(text1)
        words2 = word_tokenize(text2)
        print("WORDS1: ", words1)
        print("WORDS2: ", words2)
        return self._calculate_matrix(words1, words2, method)
    
    def _match_sentences(self, text1, text2, method) -> List[MatchResult]:
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        return self._calculate_matrix(sentences1, sentences2, method)

    def _match_full_text(self, text1, text2, method) -> List[MatchResult]:
        score = self._calculate_score(text1, text2, method)
        return [MatchResult(text1_segment=text1, text2_segment=text2, score=score)] if score >= self.threshold else []

    def _calculate_matrix(self, list1, list2, method) -> List[MatchResult]:
        results = []
        for item1 in list1:
            for item2 in list2:
                score = self._calculate_score(item1, item2, method)
                if score >= self.threshold:
                    results.append(MatchResult(text1_segment=item1, text2_segment=item2, score=score))
        # Sort and return top_k results
        results = sorted(results, key=lambda x: x.score, reverse=True)[:self.top_k]
        return results

    def _calculate_score(self, text1, text2, method):
        if method == "embedding":
            return self._embedding_score(text1, text2)
        elif method == "tfidf":
            return self._tfidf_score(text1, text2)
        elif method == "bm25":
            return self._bm25_score(text1, text2)
        else:
            raise ValueError("Invalid type specified. Choose from 'embedding', 'tfidf', or 'bm25'.")

    def _embedding_score(self, text1, text2):
        query_embedding = self.embedding_model.encode(text1, normalize_embeddings=self.normalize)
        doc_embedding = self.embedding_model.encode(text2, normalize_embeddings=self.normalize)
        score = util.cos_sim(query_embedding, doc_embedding).item()
        return score

    def _tfidf_score(self, text1, text2):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        score = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray().item()
        return score

    def _bm25_score(self, text1, text2):
        tokenized_docs = [text2.split(" ")]
        bm25 = BM25Okapi(tokenized_docs)
        score = bm25.get_scores(text1.split(" "))[0]
        return score