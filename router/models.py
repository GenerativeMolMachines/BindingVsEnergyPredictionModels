from pydantic import BaseModel, Field
from typing import Optional
from fastapi import HTTPException
from fastapi.responses import Response
import httpx


class PeptideRequest(BaseModel):
    cell_line: Optional[str] = ""
    seed: int = Field(default=42, ge=0)
