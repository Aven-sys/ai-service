from pydantic import BaseModel
from typing import List

# Model for each sentence pair
class SentencePair(BaseModel):
    text1: str
    text2: str

# Wrapper model for the list of sentence pairs
class ListRerankRequestDto(BaseModel):
    pairs: List[SentencePair]
    model_name: str 
    sorted: str