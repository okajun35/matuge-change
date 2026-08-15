"""Browser-safe runtime configuration."""

from __future__ import annotations

from fastapi import APIRouter

from backend.infrastructure import supabase_gateway

router = APIRouter(prefix="/api")


@router.get("/config")
async def get_config():
    return {"supabase": supabase_gateway.public_config()}
