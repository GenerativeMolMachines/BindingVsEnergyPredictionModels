from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from service import predict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/aptamer_mol_binding")
@limiter.limit("121/minute")
async def aptamer_mol_binding(
        request: Request,
        rna_mol_smiles: str
):
    res = {}
    seq_pair_list = rna_mol_smiles.split(";")

    if len(seq_pair_list) > 502:
        error_text = "The number of sequences in the query exceeds 500"
        raise HTTPException(status_code=429, detail=error_text)

    for seq_pair in seq_pair_list:
        ss = seq_pair.split(">")
        rna_sequences = ss[0]
        mol_smiles = ss[1]
        try:
            # predict(rna_sequences, mol_smiles)
            ans = predict(rna_sequences, mol_smiles)
        except:
            ans = None
        res[seq_pair] = ans

    return {"result": res}
