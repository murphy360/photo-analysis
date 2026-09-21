from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import analyze, health
from app.core.db import init_db
from app.pipeline import triage


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    triage.load_model()
    yield


app = FastAPI(title="Photo Analysis Service", lifespan=lifespan)

app.include_router(health.router)
app.include_router(analyze.router)

# Manual test console for dragging in a photo and seeing the actual pipeline
# output (triage objects, tier decision, people, description) rather than just
# a 202: http://localhost:8000/ui. Mounted after the API routers, and under
# its own prefix, so it can never shadow a real endpoint.
app.mount(
    "/ui",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="ui",
)
