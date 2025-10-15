from fastapi import FastAPI, Query, HTTPException
from starlette.requests import Request

from for_one_chain import predict_affinity_one_chain
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/predict_affinity_LH")
@limiter.limit("121/minute")
async def predict_affinity_LH(
        request: Request,
        anitibody_antigen: str = Query(default="")
):
    res = {}
    seq_pair_list = anitibody_antigen.split(";")
    print(len(seq_pair_list))
    if len(seq_pair_list) > 502:
        error_text = "The number of sequences in the query exceeds 500"
        raise HTTPException(status_code=429, detail=error_text)

    for seq_pair in seq_pair_list:
        if seq_pair == "":
            continue
        ss = seq_pair.split(">")
        anitibody = ss[0]
        antigen = ss[1]
        try:
            ans = predict_affinity_one_chain(Hchain=anitibody, antigen=antigen)
        except:
            ans = None
        res[seq_pair] = ans

    return {"result": res}
