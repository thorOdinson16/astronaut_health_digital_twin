"""
api/routes/config.py

Exposes non-sensitive runtime configuration to the frontend.
The Groq API key is read from the server environment (.env) so it
never has to be hard-coded in the HTML file.
"""

import os
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
async def get_frontend_config():
    """
    Returns public runtime config the frontend needs.
    Only called once at page load — keeps secrets off the client source.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on the server."
        )
    return {"groq_api_key": groq_key}