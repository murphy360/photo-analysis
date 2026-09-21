# photo-analysis

Standalone Docker microservice that analyzes photos and videos for a home
automation suite: local object-detection triage, CompreFace facial
recognition ("who"), and multi-provider vision-LLM scene description ("what's
happening"), with per-camera cost policy so a routine backyard deer doesn't
burn LLM tokens. A thin MCP adapter sits alongside it so Claude (or any other
MCP client) can query the same data conversationally.

## Architecture

```
Reolink camera --(FTP)--> Home Assistant --(folder-watch automation)--┐
                                                                       │  POST /v1/analyze
memoire, curl, etc. ---------------------------------------------------┤  (multipart file + source)
                                                                       ▼
                                                          ┌─────────────────────┐
                                                          │   FastAPI service   │
                                                          │  (this repo, app/)  │
                                                          └──────────┬──────────┘
                                                                     │ 202 + job id, background task
                                                                     ▼
                                                     ┌───────────────────────────────┐
                                                     │ 1. triage (YOLOv8n, free)      │
                                                     │ 2. policy: decide tier         │
                                                     │ 3a. CompreFace (who)           │  concurrent
                                                     │ 3b. vision-LLM(s) (what)       │
                                                     │ 4. persist + fire callback_url │
                                                     └───────────────────────────────┘
```

There is exactly one place business logic lives: the REST API in `app/`.
Home Assistant, memoire, curl, and the MCP adapter (`mcp_server/`) are all
just clients of it — that's what keeps it reusable across projects (req #5).

### Cost control (the "don't burn tokens on a deer" problem)

Every image gets a **free, local, always-on** first pass — YOLOv8n
(`app/pipeline/triage.py`) — which detects objects and buckets them into
`person` / `animal` / `vehicle` / `other`. No API key, no cost, runs on every
single image, before any paid provider is even considered.

**Known limitation:** YOLOv8n's classes come from COCO, which has no "deer"
(or most wildlife) class, so a backyard deer typically gets reported as one
of the closest quadruped classes (dog/horse/cow/sheep/bear) rather than
matched exactly. That's fine for cost control — it still buckets correctly
as `animal`, which is all the policy below needs — but the specific label in
a `skip`-tier result can be wrong (you may see `animal (horse)`).
[MegaDetector](https://github.com/microsoft/Pytorch-Wildlife) is trained
specifically on camera-trap imagery and would classify this correctly, and
was the original plan for this triage step, but its current PyPI package
pulls in a broken, unrelated dependency chain (an eagerly-imported
bioacoustics submodule requiring `librosa`/`soundfile`, and a bundled legacy
`yolov5` package that fails outright on `pkg_resources`/setuptools not being
preinstalled) — not something worth depending on for infrastructure this
central until that's fixed upstream. Revisit if/when it's worth the swap.
[SpeciesNet](https://github.com/google/cameratrapai) would be the natural
next step after that, to get actual species names (e.g. "white-tailed deer")
instead of just the `animal` bucket — its Python API isn't reliably
documented publicly either (only its CLI is), so that's also a "verify
before wiring up" task rather than something to guess at. Per-individual
recognition ("is this the same doe as yesterday") is a much harder, mostly
research-grade problem and isn't planned.

`app/config/sources.yaml` then maps what triage found, plus the camera
("source") it came from, to an **analysis tier**:

| Tier | What happens |
|---|---|
| `skip` | Triage result only. No face-id, no LLM call. |
| `cheap` | One fast/cheap vision-LLM call. |
| `standard` | One capable vision-LLM call. |
| `thorough` | 2+ providers, cross-checked (same generator/verifier idea as trivia_service). Face-id always runs, even if triage missed a person. |

Each source also has a `max_daily_analyses` hard cap — once hit, everything
falls back to `skip` regardless of what triage sees, so a stuck camera can't
run up a bill. See `app/config/sources.example.yaml` for the full policy
format and a worked example (yard cam vs. front door vs. memoire uploads).


## Running it

```bash
cp .env.example .env
# edit .env: at least one vision-LLM provider key, CompreFace URL/key, and PHOTO_SERVICE_API_KEY
docker compose up --build
```

This starts two services:
- `app` on `:8000` — the REST API
- `mcp` on `:8001` — the MCP adapter (streamable-HTTP), for MCP clients like Claude

## API

All endpoints (except `/health`) require an `X-API-Key` header matching
`PHOTO_SERVICE_API_KEY`.

### `POST /v1/analyze`

Multipart form:
- `file` — image or video
- `source` *(required)* — camera/location identifier, e.g. `front_yard`,
  `front_door`, `driveway`, or a project name like `memoire` for non-camera
  uploads. Drives the cost policy above and later history queries.
- `tier` *(optional)* — override the policy-decided tier for this one request
  (`skip` / `cheap` / `standard` / `thorough`)
- `callback_url` *(optional)* — POSTed the finished job JSON when analysis
  completes, so callers don't have to poll (e.g. an HA webhook that turns it
  into a notification)
- `metadata` *(optional)* — JSON object, stored as-is and echoed back (e.g. a
  memoire object id)

Returns `202` with the created job immediately; analysis runs in the
background.

### `GET /v1/jobs/{id}`

Fetch one job: status, triage objects, identified people, description,
providers used, tier, budget notes.

### `GET /v1/analyses?source=front_door&since_hours=24&limit=50`

History query — e.g. "who was at the front door today."

## Home Assistant integration

HA owns the FTP folder watch; this service stays a pure API. Add a
`rest_command` and trigger it from a folder-watcher automation:

```yaml
rest_command:
  analyze_photo:
    url: "http://photo-analysis:8000/v1/analyze"
    method: POST
    headers:
      X-API-Key: !secret photo_service_api_key
    payload: >
      {{ {'source': trigger.event.data.path.split('/')[-2]} }}
    content_type: "multipart/form-data"
```

(Exact automation trigger/payload wiring depends on your folder-watch setup —
the key point is HA reads the file and POSTs it, with `source` set to the
camera's location.)

## CompreFace

Point `COMPREFACE_URL` / `COMPREFACE_RECOGNITION_API_KEY` at your existing
CompreFace deployment's recognition service. Identity is entirely
CompreFace's — this service doesn't maintain its own face gallery.

## MCP adapter

`mcp_server/` exposes three tools over streamable-HTTP, each just calling the
REST API above: `get_analysis`, `list_recent_analyses`, `analyze_photo_url`.
Point an MCP client at `http://<host>:8001` to ask things like "who was at
the front door today" conversationally.

## Providers

Vision-LLM providers (`app/providers/`) are enabled purely by whether their
API key is set — Anthropic, Google Gemini, OpenAI, and xAI Grok (which speaks
the OpenAI-compatible API, so it reuses the OpenAI SDK against a different
base URL). Add a new one by writing an adapter matching `VisionProvider` in
`app/providers/base.py` and registering it in `app/providers/registry.py`.

## Development

```bash
docker compose build app
docker compose run --rm app pytest -v
```
