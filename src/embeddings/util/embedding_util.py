from sentence_transformers import SentenceTransformer
import sys
import json
import numpy as np
from embeddings.payload.response.embedding_response_dto import EmbeddingResponseDto

def generate_embeddings(input_string, model_name, normalize=False):
    model = SentenceTransformer(model_name)
    # Ensure input_string is always a list
    if isinstance(input_string, str):
        input_string = [input_string]
    embeddings = model.encode(input_string)  # Generates embeddings for the input
    if normalize:
        if embeddings.ndim == 1:  # Single embedding vector
            # Normalize the single vector
            norm = np.linalg.norm(embeddings)
            embeddings = embeddings / norm
        elif embeddings.ndim == 2:  # Matrix of embeddings
            # Normalize each vector in the matrix
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

    embedding_resposne_dto = EmbeddingResponseDto(embedding=embeddings[0].tolist())
    return embedding_resposne_dto