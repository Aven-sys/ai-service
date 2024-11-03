from pydantic import BaseModel
from typing import List

# Model for each sentence pair
class rerank_entity(BaseModel):
    text: str
    key_id: int

# Wrapper model for the list of sentence pairs
class ListSingleRerankRequestDto(BaseModel):
    main_text: str
    entity_list: List[rerank_entity]
    model: str 
    sorted: str