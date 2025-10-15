from fastapi import FastAPI, Query, HTTPException
from starlette.requests import Request

from service import predict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/mfe_rna_rna")
@limiter.limit("121/minute")
async def mfe_rna_rna(
        request: Request,
        rna1_rna2: str = Query(default="")
):
    res = {}
    seq_pair_list = rna1_rna2.split(";")

    if len(seq_pair_list) > 502:
        error_text = "The number of sequences in the query exceeds 500"
        raise HTTPException(status_code=429, detail=error_text)

    for seq_pair in seq_pair_list:
        ss = seq_pair.split(">")
        rna1_sequences = ss[0]
        rna2_sequences = ss[1]
        try:
            ans = predict(rna1_sequences, rna2_sequences)
        except:
            ans = None
        res[seq_pair] = ans
    return {"result": res}
