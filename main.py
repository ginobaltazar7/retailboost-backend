'''
  Proof of concept main() for RetailBoost
'''

# import needed modules
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.startup import init_task_status_table
from api.endpoints import router as api_router
from dotenv import load_dotenv

load_dotenv()  # take variables from .env


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_task_status_table()
    yield


app = FastAPI(lifespan=lifespan)

# Configure CORS (Add other allowed origins through '.env' file)
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, etc.
    allow_headers=["*"],  # Allows Content-Type, Authorization, etc.
)

app.include_router(api_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
