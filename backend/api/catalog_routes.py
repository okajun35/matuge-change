"""Product lash catalog endpoints (registration, browsing, similarity search)."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Form
from fastapi.responses import Response

from backend.api.container import container
from backend.api.errors import to_http
from backend.catalog.asset import AssetDraft
from backend.sessions.errors import MatteNotReady

router = APIRouter(prefix="/api/assets")


@router.post("")
async def register_asset(
    session_id: str = Form(...),
    name: str = Form(...),
    brand: str | None = Form(None),
):
    app = container()
    try:
        app.store.require(session_id)
        if not app.store.has_layer(session_id, "product_rgba"):
            raise MatteNotReady("run matting first")
        roi = app.sessions.roi_of(session_id)
        draft = AssetDraft(
            name=name,
            rgba=app.store.load_image(session_id, "product_rgba", flags=-1),
            roi=(roi.x0, roi.y0, roi.x1, roi.y1),
            roi_scale=roi.scale,
            landmarks=app.store.load_array(session_id, "landmarks"),
            brand=brand,
            session_id=session_id,
        )
        asset = app.catalog.register(draft)
    except Exception as exc:
        raise to_http(exc) from exc
    return {
        "id": asset.id,
        "name": asset.name,
        "brand": asset.brand,
        "width": asset.width,
        "height": asset.height,
        "alpha_coverage": asset.alpha_coverage,
        "storage_path": asset.storage_path,
    }


@router.get("")
async def list_assets(limit: int = 12, offset: int = 0, q: str = ""):
    page = container().catalog.list_assets(limit=limit, offset=offset, query=q)
    return {"assets": page["items"], "total": page["total"], "limit": limit, "offset": offset}


@router.get("/{asset_id}")
async def get_asset(asset_id: str):
    try:
        row = dict(container().catalog.get_asset(asset_id))
    except Exception as exc:
        raise to_http(exc) from exc
    row.pop("embedding", None)
    row.pop("landmarks", None)
    return row


def _disposition(kind: str, asset_id: str, filename: str) -> dict[str, str]:
    """Content-Disposition with a UTF-8 filename (product names are Japanese)."""
    name = quote(filename)
    return {"Content-Disposition": f"{kind}; filename=\"{asset_id}.png\"; filename*=UTF-8''{name}"}


def _asset_filename(asset_id: str, suffix: str) -> str:
    try:
        name = str(container().catalog.get_asset(asset_id).get("name") or asset_id)
    except Exception:
        name = asset_id
    return f"{name}{suffix}.png"


@router.get("/{asset_id}/image")
async def get_asset_image(asset_id: str):
    """Product RGBA PNG (alpha preserved). Inline so the catalog can show it in <img>."""
    try:
        png = container().catalog.load_png(asset_id)
    except Exception as exc:
        raise to_http(exc) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers=_disposition("inline", asset_id, _asset_filename(asset_id, "")),
    )


@router.get("/{asset_id}/mask")
async def get_asset_mask(asset_id: str):
    """Alpha channel only, as a grayscale PNG download."""
    try:
        png = container().catalog.load_mask_png(asset_id)
    except Exception as exc:
        raise to_http(exc) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers=_disposition("attachment", asset_id, _asset_filename(asset_id, "-mask")),
    )


@router.get("/{asset_id}/similar")
async def similar_assets(asset_id: str, limit: int = 5):
    try:
        return {"matches": container().catalog.find_similar(asset_id, limit)}
    except Exception as exc:
        raise to_http(exc) from exc
