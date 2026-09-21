from contextlib import asynccontextmanager

from fastapi import FastAPI

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
