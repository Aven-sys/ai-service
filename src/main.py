from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Get routes
from embeddings.router import embedding_router
from reranker.router import reranker_router
from llm.router import llm_router
from preprocess.router import preprocess_router
from job_candidate.router import job_candidate_router
from Interview_llm.router import interview_llm_router

# Initialize Fast API
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allows all origins
    allow_credentials=True,    # Allows cookies to be included in cross-origin requests
    allow_methods=["*"],       # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],       # Allows all headers
)

# Include routers
app.include_router(embedding_router.router)
app.include_router(reranker_router.router)
app.include_router(llm_router.router)
app.include_router(preprocess_router.router)
app.include_router(job_candidate_router.router) 
app.include_router(interview_llm_router.router)

@app.get("/")
async def first_api():
    return {"message": "Hello World"}
