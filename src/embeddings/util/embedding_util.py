from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from embeddings.payload.response.embedding_response_dto import EmbeddingResponseDto

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name).to(device)

def generate_embeddings(input_string, model_name, normalize=False):
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