# photo-analysis
Docker Container that uses various tools to analyze photos contextually

## Run locally

```bash
uvicorn main:app --reload
```

## Run with Docker

```bash
docker build -t photo-analysis .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=your_key photo-analysis
```
