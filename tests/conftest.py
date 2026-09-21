import os
import tempfile

# Force (not setdefault): docker-compose loads .env into the container's real
# environment before pytest runs, so these must override it unconditionally —
# otherwise tests silently point at the real API key and the real persistent DB.
os.environ["PHOTO_SERVICE_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tempfile.mktemp(suffix='.db')}"

import pytest_asyncio  # noqa: E402

from app.core.db import init_db  # noqa: E402
from app.pipeline.types import DescriptionResult  # noqa: E402


class FakeProvider:
    """A VisionProvider stand-in with no network calls, for pipeline/API tests."""

    def __init__(self, name: str, text: str = "A person walks up to the door.") -> None:
        self.name = name
        self._text = text
        self.calls = 0

    async def describe(
        self, image_bytes: bytes, mime_type: str, *, cheap: bool = False
    ) -> DescriptionResult:
        self.calls += 1
        return DescriptionResult(text=self._text, provider=self.name)


@pytest_asyncio.fixture(autouse=True)
async def _init_db():
    await init_db()
    yield
