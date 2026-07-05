# Track & Treat — Recommendation Service

Deterministic content-based-filtering (CBF) microservice that generates 7-day
meal plans. NestJS calls it over HTTP; it is the only component that touches the
model bundle. No LLM is involved in plan generation.

## Endpoints
- `GET /health` — returns `200` only once artifacts are loaded (Docker health-gates on this).
- `POST /generate-plan` — body `{ userProfile, adaptiveProfile }` → a 7-day plan.

## Artifacts (loaded on startup)
Live in the backend repo at `track-treat/artifacts/` (single canonical source,
also seeded into the DB). Override the location with `ARTIFACTS_DIR`.

| File | Contents |
|---|---|
| `cbf_model.pkl` | meal matrix + `meal_ids`/`meal_names` + feature schema + fitted params (bounds, category maps, allergen map, category prefs, tag thresholds) |
| `meal_matrix.npy` | `n × 28` float32 meal vectors (also inside the pkl) |
| `normalization_bounds.json` | `max_calories`, `max_fiber`, `max_prep` |
| `meal_catalog.json` | per-meal composition (ingredient + grams), meal_type, complexity, prep |
| `ingredient_table.json` | per-100g nutrition + USDA category for each ingredient |

## Run (dev)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py            # http://localhost:5001
.venv/bin/python -m pytest tests/  # skew + allergen + plan-invariant suites
```

## Run (Docker)
Wired into `track-treat/docker-compose.yml`:
```bash
cd ../track-treat && docker compose up -d recommendation-service
```
The backend reaches it via `RECOMMENDATION_SERVICE_URL` (default `http://localhost:5001`).

## Correctness guarantees
- **No train/serve skew:** `tests/validate_skew.py` rebuilds every catalog meal
  vector with the shared `vectorization.py` and asserts it equals the shipped
  `meal_matrix.npy`. The service refuses to start if `feature_schema` drifts.
- **Allergen safety:** every allergen id must exclude the meals containing its
  ingredients; unknown ids are rejected, not silently ignored.
- **Deterministic assembly:** slots, ±10% portion scaling, ≤2 repeats/week, and
  vegan/vegetarian as a hard filter (never serve animal products to a vegan).

## Regenerating the model (offline)
The artifacts are an immutable, versioned snapshot. To change what's
recommendable (add an ingredient or meal, retune a category):

1. Update the canonical dataset and re-run Notebooks 1–3 (offline).
2. Replace the five files in `track-treat/artifacts/`.
3. Re-seed the DB so ids stay aligned: `cd ../track-treat && pnpm seed`.
4. Restart the service. `artifactVersion` (a hash of `cbf_model.pkl`) changes
   automatically and is stamped onto every plan generated thereafter.

Never mutate `meal_matrix` or the catalog live — a plan must stay reproducible
from its `artifactVersion`.
