from fastapi import FastAPI, Query, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

from service import predict

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/link_prediction")
@limiter.limit("121/minute")
async def link_prediction(
        request: Request,
        target_id: str
):
    if len(target_id) < 1:
        error_text = "ID is invalid"
        raise HTTPException(status_code=404, detail=error_text)
    try:
        return {"result": predict(target_id)}
    except:
        raise HTTPException(status_code=422)
