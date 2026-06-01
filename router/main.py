from fastapi import FastAPI, Query, HTTPException
import httpx

app = FastAPI(
    root_path="/api"
)

# Внутренние адреса контейнеров в Docker-сети
RNA_RNA_URL = "http://rna_viennarna_model:4425/mfe_rna_rna"
APT_PROT_URL = "http://apt_prot_model:4426/aptamer_prot_binding"


@app.post("/mfe_rna_rna")
async def mfe_rna_rna(
    rna1_rna2: str = Query(default="")
):
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                RNA_RNA_URL,
                params={
                    "rna1_rna2": rna1_rna2
                }
            )

        return response.json()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RNA-RNA service unavailable: {str(e)}"
        )


@app.post("/aptamer_prot_binding")
async def aptamer_prot_binding(
    sequences: str
):
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                APT_PROT_URL,
                params={
                    "sequences": sequences
                }
            )

        return response.json()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Aptamer-Protein service unavailable: {str(e)}"
        )
