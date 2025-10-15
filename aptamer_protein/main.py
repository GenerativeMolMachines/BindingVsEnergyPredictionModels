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


@app.post("/aptamer_prot_binding")
@limiter.limit("121/minute")
async def aptamer_prot_binding(
        request: Request,
        sequences: str
):
    res = {}
    seq_pair_list = sequences.split(";")

    if len(seq_pair_list) > 502:
        error_text = "The number of sequences in the query exceeds 500"
        raise HTTPException(status_code=429, detail=error_text)

    for seq_pair in seq_pair_list:
        if seq_pair == "":
            continue
        ss = seq_pair.split(">")
        apt_sequences = ss[0]
        prot_sequences = ss[1]
        try:
            # predict(apt_sequences, prot_sequences)
            ans = predict(apt_sequences, prot_sequences)
        except:
            ans = None
        res[seq_pair] = ans

    return {"result": res}
