from fastapi import FastAPI

# Get routes
from embeddings.router.embedding_router import router as embedding_router

app = FastAPI()
app.include_router(embedding_router)


@app.get("/")
async def first_api():
    return {"message": "Hello World"}
