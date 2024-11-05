from pydantic import BaseModel
from typing import Optional

# Wrapper model for the list of sentence pairs
class PreprocessTextRequestDto(BaseModel):
    text: str
    remove_stopwords: Optional[bool] = False
    remove_punctuation: Optional[bool] = False
    remove_special_characters: Optional[bool] = False
    lowercase: Optional[bool] = False
    lemmatize: Optional[bool] = False