"""
DubbingStudio - FastAPI application entry point.
Serves the REST API, WebSocket endpoints, and static frontend files.
"""
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import init_db
from app.api.jobs import router as jobs_router
from app.api.websocket import router as ws_router
from app.api.system import router as system_router
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    await init_db()
    logger.info("DubbingStudio ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="DubbingStudio API",
    description="AI-powered video transcription, translation, and voice-cloned dubbing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(ws_router)
app.include_router(system_router)

# Serve React frontend (production build)
frontend_path = Path(settings.frontend_dir)
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_path / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str = ""):
        index = frontend_path / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "DubbingStudio API is running. Build the frontend to serve the UI."}
else:
    @app.get("/", include_in_schema=False)
    async def api_root():
        return {
            "message": "DubbingStudio API running",
            "docs": "/docs",
            "note": "Run `npm run build` in the frontend directory to serve the UI.",
        }
