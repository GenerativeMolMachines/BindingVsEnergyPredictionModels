import httpx
from fastapi import FastAPI, Query, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request as StarletteRequest

# Настройка rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Конфигурация внутренних сервисов
INTERNAL_SERVICES = {
    "mfe_rna_rna": "http://rna_viennarna_model:4425/mfe_rna_rna",
    "aptamer_prot_binding": "http://aptamer_protein-app:4426/aptamer_prot_binding"
}

# Таймаут для запросов
REQUEST_TIMEOUT = 30.0


@app.post("/mfe_rna_rna")
@limiter.limit("121/minute")
async def mfe_rna_rna(
        request: Request,
        rna1_rna2: str = Query(default="CC>CC;")
):
    """Прокси запрос к внутреннему сервису mfe_rna_rna"""

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            internal_url = INTERNAL_SERVICES["mfe_rna_rna"]

            response = await client.post(
                internal_url,
                params={"rna1_rna2": rna1_rna2},
                headers={
                    "X-Forwarded-For": request.client.host if request.client else "unknown",
                    "X-Real-IP": request.client.host if request.client else "unknown"
                }
            )

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Gateway Timeout: Internal service did not respond in time"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json() if e.response.text else str(e)
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Bad Gateway: Could not connect to internal service - {str(e)}"
            )


@app.post("/aptamer_prot_binding")
@limiter.limit("121/minute")
async def aptamer_prot_binding(
        request: Request,
        sequences: str = Query(default="CC>AMC;")
):
    """Прокси запрос к внутреннему сервису aptamer_prot_binding"""

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            internal_url = INTERNAL_SERVICES["aptamer_prot_binding"]

            response = await client.post(
                internal_url,
                params={"sequences": sequences},
                headers={
                    "X-Forwarded-For": request.client.host if request.client else "unknown",
                    "X-Real-IP": request.client.host if request.client else "unknown"
                }
            )

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Gateway Timeout: Internal service did not respond in time"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json() if e.response.text else str(e)
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Bad Gateway: Could not connect to internal service - {str(e)}"
            )
