from ..util import embedding_util

def get_embedding():
    return embedding_util.generate_embeddings("This is a test", "all-mpnet-base-v2", True)
