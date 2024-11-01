from pydantic import BaseModel

class EmbeddingRequestDto(BaseModel):
    text: str
    model: str
    normalize: bool

    