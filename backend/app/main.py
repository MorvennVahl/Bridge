"""FastAPI entry point for the Bridge backend.

Serves /api/* endpoints and, in a Modal deploy, mounts the built React SPA at
the root path.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.data import data_files_present
from backend.app.models import HealthResponse
from backend.app.routers import annotations, catalog, compute, predict, swarm, tables, workflow

app = FastAPI(title="Bridge API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(catalog.router)
app.include_router(predict.router)
app.include_router(compute.router)
app.include_router(annotations.router)
app.include_router(tables.router)
app.include_router(workflow.router)
app.include_router(swarm.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", data_files_present=data_files_present())


# Mount built SPA if present (only true in Modal deploy after `npm run build`).
REPO_ROOT = Path(__file__).resolve().parents[2]
DIST = REPO_ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def spa_root() -> FileResponse:
        return FileResponse(DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        target = DIST / full_path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
