"""Thin MCP adapter over the photo-analysis REST API.

Deliberately has no business logic of its own — every tool just calls the same
REST API that Home Assistant and other clients use (app/api/analyze.py), so
there is exactly one place cost policy, auth, and the analysis pipeline live.
This just gives MCP clients (e.g. Claude) a conversational way to query it:
"who was at the front door today", "analyze this photo", etc.
"""

import os
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

API_URL = os.environ.get("PHOTO_API_URL", "http://app:8000").rstrip("/")
API_KEY = os.environ.get("PHOTO_API_KEY", "")

mcp = MCPServer("photo-analysis")


def _headers() -> dict[str, str]:
    return {"x-api-key": API_KEY}


async def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{API_URL}{path}", headers=_headers(), params=params)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_analysis(job_id: int) -> dict:
    """Fetch a single photo/video analysis job by id: its status, the objects
    local triage detected, any people CompreFace identified, and the scene
    description."""
    return await _get(f"/v1/jobs/{job_id}")


@mcp.tool()
async def list_recent_analyses(
    source: str | None = None,
    since_hours: int = 24,
    limit: int = 20,
) -> list[dict]:
    """List recent analyses, optionally filtered by camera/location ('source',
    e.g. 'front_door', 'driveway') and how far back to look. Use this to answer
    things like 'who was at the front door today' or 'what happened on the
    driveway camera last night'."""
    params: dict[str, Any] = {"since_hours": since_hours, "limit": limit}
    if source:
        params["source"] = source
    return await _get("/v1/analyses", params=params)


@mcp.tool()
async def analyze_photo_url(image_url: str, source: str, tier: str | None = None) -> dict:
    """Trigger analysis of a photo at a URL for the given source (a camera/
    location name, or a project name like 'memoire' for ad-hoc uploads).
    Downloads the image and submits it to the analysis service; returns the
    created job — call get_analysis with its id to see the result once the
    background analysis finishes."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        image_response = await client.get(image_url)
        image_response.raise_for_status()
        content_type = image_response.headers.get("content-type", "image/jpeg")

        files = {"file": ("image", image_response.content, content_type)}
        data = {"source": source}
        if tier:
            data["tier"] = tier
        response = await client.post(
            f"{API_URL}/v1/analyze", headers=_headers(), files=files, data=data
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
