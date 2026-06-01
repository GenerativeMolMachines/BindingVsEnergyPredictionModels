import httpx
import httpx
from fastapi import FastAPI, Query, HTTPException
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from models import PeptideRequest

app = FastAPI(
    root_path="/api"
)

# Внутренние адреса контейнеров в Docker-сети
RNA_RNA_URL = "http://rna_viennarna_model:4425/mfe_rna_rna"
APT_PROT_URL = "http://apt_prot_model:4426/aptamer_prot_binding"
APT_MOL_URL = "http://apt_mol_model:4431/aptamer_mol_binding"
PROT_PROT_URL = "http://protein_protein_cont:4418/protein_protein_binding"
PEPTIDE_URL = "http://cpp_gen_model:4397/generate_peptide"


@app.post("/mfe_rna_rna")
async def mfe_rna_rna(
        rna1_rna2: str = Query(default="СС>CC;")
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
        sequences: str = Query(default="СС>AMG;")
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


@app.post("/aptamer_mol_binding")
async def aptamer_mol_binding(
        rna_mol_smiles: str = Query(default="CC>CC(O)C;")
):
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                APT_MOL_URL,
                params={
                    "rna_mol_smiles": rna_mol_smiles
                }
            )

        return response.json()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Aptamer-Molecule service unavailable: {str(e)}"
        )


@app.post("/protein_protein_binding")
async def protein_protein_binding(
        sequences: str = Query(default="AAA>BBB")
):
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                PROT_PROT_URL,
                params={
                    "sequences": sequences
                }
            )

        return response.json()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Protein-Protein service unavailable: {str(e)}"
        )


@app.post("/generate_peptide")
async def generate_peptide(payload: PeptideRequest):
    try:
        request_body = {
            "cell_line": payload.cell_line,
            "seed": payload.seed,

            # параметры сервиса по умолчанию
            "prompt_text": "",
            "max_new_tokens": 50,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 0,
        }

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                PEPTIDE_URL,
                json=request_body
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get(
                "content-type",
                "application/json"
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Peptide service unavailable: {str(e)}"
        )
