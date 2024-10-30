from pydantic import BaseModel

class EmbeddingRequestDto(BaseModel):
    text: str
    model_name: str
    normalize: bool

    