from pydantic import BaseModel
from typing import List

# Wrapper model for the list of sentence pairs
class PreprocessTextResponseDto(BaseModel):
    preprocess_text: str
    postprocess_text: str