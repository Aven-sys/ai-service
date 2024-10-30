from fastapi import APIRouter

router = APIRouter(
    prefix="/embeddings",
)


@router.get("/")
async def embedding_test():
    return {"message": "Embeddings"}
