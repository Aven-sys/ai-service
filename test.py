from collections import Counter
import math

class CustomBM25:
    def __init__(self, corpus: list, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        
        # Calculate document lengths and average
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size
        
        # Calculate IDF scores for all terms
        self.idf = self._calculate_idf()
    
    def _calculate_idf(self) -> dict:
        # Count documents containing each term
        df = Counter()
        for doc in self.corpus:
            terms = set(doc)  # Count each term only once per document
            for term in terms:
                df[term] += 1
        
        # Calculate IDF for each term
        idf = {}
        for term, doc_freq in df.items():
            # Standard BM25 IDF formula
            idf[term] = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        
        return idf
    
    def get_scores(self, query: list) -> list:
        scores = []
        
        for doc_id, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_lens[doc_id]
            
            # Count term frequencies in document
            doc_terms = Counter(doc)
            
            for term in query:
                if term not in self.idf:
                    continue
                
                # Get term frequency in document
                tf = doc_terms.get(term, 0)
                
                # BM25 scoring formula
                numerator = self.idf[term] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += (numerator / denominator) if denominator != 0 else 0
            
            scores.append(score)
        
        return scores

def test_bm25():
    # Test corpus (documents to search in)
    corpus = [
        "This sample text is meant to test the similarity matcher functionality.",
        "This is when i say Hello world.",
        "Another completely different document about cats and dogs.",
        "This is a test document that has some similar words.",
        "Hello world this is a programming example."
    ]
    
    # Test queries
    queries = [
        "This is a sample text for similarity matching.",
        "This is when i say Hello world.",
        "Something about cats",
        "test document similar"
    ]
    
    # Tokenize corpus and queries
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_queries = [query.lower().split() for query in queries]
    
    # Initialize BM25
    bm25 = CustomBM25(tokenized_corpus)
    
    # Test each query
    print("Testing BM25 Scoring:\n")
    for query, query_tokens in zip(queries, tokenized_queries):
        print(f"\nQuery: {query}")
        scores = bm25.get_scores(query_tokens)
        
        # Print scores with documents
        print("\nMatches:")
        scored_docs = list(zip(corpus, scores))
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        for doc, score in scored_docs:
            print(f"\nDocument: {doc}")
            print(f"Score: {score:.4f}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    test_bm25()