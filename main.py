import io
import json
import os

import google.generativeai as genai
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


class ImageRecognitionResult(BaseModel):
    subjects: list[str] = Field(
        description="List of primary objects or people in the photo."
    )
    text_content: str = Field(
        description="Any OCR text identified (signs, documents, handwriting)."
    )
    estimated_era: str = Field(
        description="The probable decade or time period based on visual cues."
    )
    visual_summary: str = Field(
        description="A concise 2-sentence description of the scene."
    )


app = FastAPI(title="Photo Analysis Service")


@app.on_event("startup")
def configure_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    app.state.gemini_model = genai.GenerativeModel("gemini-3-flash")


@app.post("/analyze", response_model=ImageRecognitionResult)
async def analyze_image(file: UploadFile = File(...)) -> ImageRecognitionResult:
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    prompt = (
        "Analyze this photo and respond with JSON that strictly matches this schema: "
        '{"subjects": ["string"], "text_content": "string", '
        '"estimated_era": "string", "visual_summary": "string"}.\n'
        "visual_summary must be exactly 2 sentences."
    )

    try:
        response = app.state.gemini_model.generate_content(
            [prompt, image],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ImageRecognitionResult.model_json_schema(),
            },
        )
        return ImageRecognitionResult.model_validate(json.loads(response.text))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Model did not return valid JSON") from exc
    except Exception as exc:  # pragma: no cover - protects service boundary
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc
