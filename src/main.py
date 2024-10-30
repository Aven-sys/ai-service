from fastapi import FastAPI

# Get routes
from embeddings.router import embedding_router

app = FastAPI()
app.include_router(embedding_router.router)


@app.get("/")
async def first_api():
    return {"message": "Hello World"}
