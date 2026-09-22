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
("source") it came from, to an **analysis tier** — but only for the paid
vision-LLM description step. Local triage and CompreFace face-id are both
free (self-hosted, no per-call cost) and **always run, on every image,
regardless of tier** — there's no reason to gate either behind the paid-LLM
decision, and it means a person the object detector missed (partial
occlusion, small/distant, misclassified as something else) still gets a
chance at being identified. Tiers, cheapest to most expensive:

| Tier | What happens |
|---|---|
| `skip` | No paid LLM call at all. |
| `cheap` | One fast/cheap vision-LLM call. |
| `standard` | One capable vision-LLM call. |
| `thorough` | 2+ providers, cross-checked (same generator/verifier idea as trivia_service). |

Each source also has a `max_daily_analyses` hard cap — once hit, everything
falls back to `skip` regardless of what triage sees, so a stuck camera can't
run up a bill (triage and CompreFace still run; only the LLM call is capped).
See `app/config/sources.example.yaml` for the full policy format and a
worked example (yard cam vs. front door vs. memoire uploads).

**Gotcha: source matching is an exact, case-sensitive string comparison.**
If your camera/automation sends `"Front Yard"` but `sources.yaml` only has a
`front_yard:` entry, that request silently falls through to `default:`
instead of erroring — you'll get conservative fallback behavior with no
indication anything's misconfigured. Whatever string your client actually
sends is the key that has to exist in `sources.yaml`; check the `source`
field on a returned job (or in `/ui`) against your config if a camera seems
to be getting the wrong tier or policy.

## Running it

```bash
cp .env.example .env
# edit .env: at least one vision-LLM provider key, CompreFace URL/key, and PHOTO_SERVICE_API_KEY
docker compose up --build
```

This starts two services:
- `app` on `:8000` — the REST API
- `mcp` on `:8001` — the MCP adapter (streamable-HTTP), for MCP clients like Claude

## Test console (`/ui`)

`http://<host>:8000/ui` is a manual test page (`app/static/index.html`,
mounted in `app/main.py`): drag in a photo or video, pick a `source` and
optional tier override, and see the actual pipeline output — triage objects,
tier decision, CompreFace people (including a distinct badge for a
just-auto-enrolled face), and the LLM description — rather than just a `202
Accepted`. Paste your `PHOTO_SERVICE_API_KEY` into the field at the top
(stored in the browser's `localStorage`, never sent anywhere but this
service). Not part of the product API — just the fastest way to confirm a
change actually behaves the way you expect against a real image, including
recent history via the "Recent analyses" section (`GET /v1/analyses`).

A few things it shows that the raw API response leaves you to infer:
- **Bounding boxes**, drawn directly on the preview image — amber for every
  local-triage object, green/orange/blue for CompreFace people (recognized /
  unknown / just-enrolled). Uses each detection's `box` coordinates
  (`{x_min, y_min, x_max, y_max}` in the source image's own pixels) via an
  SVG overlay sized to the image's natural dimensions, so it lines up
  correctly regardless of how large the browser renders it.
- **Why People is empty**, when it is — `people_note` on the job distinguishes
  "CompreFace not configured" from "ran, found no face in this photo" (the
  single most common outcome on a real security camera — most frames don't
  have a clear face at all) from an actual CompreFace error, rather than
  collapsing all three into the same blank list.
- **"Compare every enabled provider"** checkbox — testing only, not something
  production traffic sets. Runs every configured vision-LLM provider
  regardless of the source's tier policy (which normally picks one, per
  `sources.yaml`) and shows each one's answer in its own card
  (`provider_results` on the job), so you can actually judge Anthropic vs.
  Gemini vs. OpenAI vs. Grok on the same photo instead of guessing from
  production's randomized/single-provider picks.
- **A failed provider stays visible instead of vanishing.** `provider_results`
  has one entry per provider *attempted*, success or failure — a card with a
  red border and the actual error message, not just a missing card. (This is
  the direct fix for a real outage: Grok's configured model name 404'd on
  every call, and the job just came back with an empty description and no
  indication why. `description`/`description_providers` still only ever
  draw from providers that actually succeeded.)

## API

All endpoints (except `/health` and `/ui`) require an `X-API-Key` header
matching `PHOTO_SERVICE_API_KEY`.

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
- `compare_providers` *(optional, testing only)* — run every enabled
  vision-LLM provider instead of the tier's usual pick; see `/ui` above.
  Costs one call per provider. Production callers should never set this.

Returns `202` with the created job immediately; analysis runs in the
background.

### `GET /v1/jobs/{id}`

Fetch one job: status, triage objects (each with a `box`), identified people
(each with a `box`, plus `people_note` explaining an empty list), description,
`provider_results` (one `{provider, text, error}` entry per vision-LLM
provider actually attempted — `error` is non-null and `text` is null for one
that failed, rather than it just being absent), tier, budget notes.

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
CompreFace deployment's recognition service (from CompreFace's UI: open your
Application → the **Face Recognition** service inside it → its API key is
shown there — not the admin key). Identity is entirely CompreFace's — this
service doesn't maintain its own face gallery.

Unrecognized faces (below `COMPREFACE_SIMILARITY_THRESHOLD`) are reported as
subject `"unknown"` rather than dropped, so a job still tells you a person
was there even when CompreFace can't name them.

**Auto-enrolling new faces.** Set `auto_enroll_unknown_faces: true` on a
source in `sources.yaml` to have an unrecognized face registered as a new
placeholder subject automatically, instead of just reported as `"unknown"`.
Placeholder subjects are named `"<unknown_face_label> <n>"` (e.g. `"Amazon
Driver 1"`, `"Amazon Driver 2"` — set `unknown_face_label` per source, e.g.
to track recurring delivery drivers on a front-door camera), numbered from
whatever already exists in CompreFace so it stays correct across restarts.
Once enrolled, the *next* sighting of that same person matches the
placeholder instead of creating a new one — so you periodically browse
CompreFace's own UI and rename the placeholders you recognize to real names;
this service never renames or merges subjects itself. Leave it off (the
default) for any source with a lot of foot traffic that isn't yours (a yard
cam catching mail carriers, neighbors, solicitors), or you'll fill your face
collection with one-off strangers. It can also be set on the top-level
`default:` block in `sources.yaml` to apply everywhere at once, same as any
other policy field.

**Recognized names reach the vision-LLM prompt.** CompreFace now runs
*before* the description step (not concurrently with it, the way it used
to) specifically so a recognized name can be passed into the prompt — the
description says "Cathleen Murphy walks a dog" instead of "a woman walks a
dog" whenever CompreFace already identified her. This costs a little extra
latency (the two steps used to run in parallel) in exchange for that
context actually being usable. The literal `"unknown"` placeholder is never
passed through as if it were a name; an auto-enrolled placeholder like
`"Amazon Driver 1"` is, since it's still more useful than "a man in a
uniform." Skip-tier's free auto-generated text gets the same treatment
(e.g. `"...Recognized: Cathleen Murphy."`) at no extra cost, since no LLM
call is involved there either way.

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

xAI's model lineup moves fast — `GROK_MODEL`/`GROK_MODEL_CHEAP` default to
`grok-4.7` (confirmed vision-capable via xAI's own docs; `grok-2-vision-1212`,
this repo's original default, no longer exists and 404s every call). Both
tiers point at the same model for now since a cheaper Grok model's vision
support isn't confirmed — check xAI's current docs before splitting them
back into two, rather than guessing a model name. If a provider's model name
ever goes stale like this again, it fails loud now: a failed provider shows
up in `provider_results` with its actual error instead of just vanishing
(see `/ui` above) — check there first rather than digging through container
logs.

**Which provider gets picked** (for tiers that use just one or two, i.e.
everything except `thorough`) isn't the first one listed in a source's
`provider_preference` — it's whichever eligible provider has answered the
*fewest* jobs historically (`app/repository/analyses.provider_usage_counts`,
tallied from `description_providers` across all past jobs), ties broken
randomly. `provider_preference` still acts as a hard filter (e.g. `memoire`'s
`[anthropic, gemini]` still never picks openai/grok even if configured) —
usage only decides the order *within* that allowed set. This balances usage
across providers over time instead of always favoring the same one, without
needing any state beyond what's already in the jobs table.

## Development

```bash
docker compose build app
docker compose run --rm app pytest -v
```

### Schema changes

The container runs `alembic upgrade head` on startup (Dockerfile `CMD`) —
that's what applies a schema change to an *existing* database. `init_db()`
(`app/core/db.py`) still calls SQLModel's `create_all`, but that only
creates tables that don't exist yet; it silently does nothing to a table
that already exists but is missing a column a newer model added, so it's
only a safety net for the test suite's fresh per-run SQLite file, not a
substitute for a real migration.

**Whenever you add/change a field on `AnalysisJob`, generate a migration for
it, or every existing database (including anyone's real one) keeps the old
schema and every insert starts failing:**

```bash
docker compose run --rm app alembic revision --autogenerate -m "add whatever_field"
```

Then open the generated file in `alembic/versions/` and add `import sqlmodel`
near the top — autogenerate doesn't add it, even though it references
`sqlmodel.sql.sqltypes.AutoString` for string columns (the `script.py.mako`
template now includes this import for anything generated from here on, but
double check). Verify the migration against a copy of a real database
before trusting it, not just a fresh empty one — `op.create_table` on a
table that already exists is a different failure than a missing column, and
only shows up against a database that already has data.
