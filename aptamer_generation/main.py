from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from pydantic import BaseModel
import numpy as np
import torch

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from service import predict


limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


class AptamerGenerationResponse(BaseModel):
    aptamer: str


@app.post("/generate_aptamer")
@limiter.limit("121/minute")
async def generate_aptamer(request: Request):
    """
    Generate aptamer sequence.
    No input parameters required.
    """
    try:
        embedding = np.random.randn(
            80,
            2048
        ).astype(np.float32)

        aptamer = predict(embedding)

        return AptamerGenerationResponse(
            aptamer=aptamer
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
