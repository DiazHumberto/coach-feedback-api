# Coach Feedback API — Demo Pipeline (Python)

This repo contains an end-to-end **demo pipeline** that can run in two modes:

1) **Annotation JSON mode** (from the Video Annotation / labeling platform)  
2) **Session-output JSON mode** (a “full analysis” JSON produced by your upstream pipeline)

In both cases the pipeline can:
- **Parse input JSON**
- **Compute / extract key metrics**
- **Apply thresholds + rubric** to produce a simple **scorecard**
- *(Optional)* call the **OpenAI API** to generate **mat-safe coach feedback**
- *(Optional)* log the full prompt + response to **LangSmith** for supervisor review

---

## Input types

### A) Annotation JSON (label export)
Expected structure: a `labels` list containing:

**Positional labels**
- `Ball Position` (frame + x,y)
- `Corner 1 of Mat`, `Corner 2 of Mat`, `Corner 3 of Mat`, `Corner 4 of Mat`
  - Corner order must be:
    1 = top-left, 2 = top-right, 3 = bottom-right, 4 = bottom-left

**Temporal labels (startFrame/endFrame)**
- `Left Ball touch`, `Right Ball touch`
- `Toe tap event`
- Optional (if your labeling includes them): `Look down`, `Ball out of frame`, `Ball out of reach`

> If an optional label doesn’t exist, related metrics default gracefully.

### B) Session-output JSON (analysis export)
This is the richer “session output” schema (ball control, balance, head up, etc.).  
The pipeline will **condense** that output into a cleaner, “ideal” set of fields (so it’s not full of duplicates / derivable values).

---

## How the pipeline works (high level)

### Step 1 — Load JSON
- Detects/uses an input type:
  - Annotation export → computes metrics from labels
  - Session-output → extracts + condenses metrics

### Step 2 — Normalize ball coordinates (annotation mode)
- Uses the 4 mat corners to compute a homography
- Converts pixel (x,y) → normalized mat coordinates (u,v) in (0..1)
- Makes speed/control metrics comparable across camera angles

### Step 3 — Metrics
- Annotation mode computes demo metrics such as:
  - touches, touches/min, toe taps/min
  - L/R balance
  - rhythm steadiness
  - ball speed + “spikiness”
  - ball control proxy score

- Session-output mode condenses to the key “ideal” fields per section

### Step 4 — Scoring (thresholds + rubric)
- Applies starter thresholds (age/skill aware where possible)
- Produces a **0–100 scorecard**
- Adds rubric **bands**:
  - 1–20, 21–40, 41–60, 61–80, 81–100 (with band labels)

### Step 5 — (Optional) LLM coach feedback
- Builds a strict prompt:
  - **ONLY mat-safe drills** (4x4 feet)
  - no jogging/passing/shooting/long-space dribbling
  - age-appropriate
- Returns structured JSON feedback:
  - `praise`
  - `strengths`
  - `improvements`
  - `drill`
  - `next_goal`

### Step 6 — (Optional) LangSmith tracing
- When enabled, the OpenAI call is traced so your supervisor can see:
  - system prompt
  - user prompt (metrics + scorecard)
  - model output
  - latency / tokens

---

## Where to edit rules (the “knobs”)
These are the most important places to modify behavior:

1) **Mat-safe drill whitelist**
- `MAT_SAFE_DRILLS`  
Add/remove drills per age band. Keeps output grounded.

2) **Coach style templates**
- `STYLE_HINTS`  
Controls tone + structure (`cheer`, `mentor`, `performance`, etc.).

3) **Hard constraints**
- `system = (...)`  
Your guardrails: mat-only, no running/passing/shooting, safe/age-appropriate.

4) **Thresholds + rubric**
- scoring dicts / functions (`THRESHOLDS`, `RUBRIC_BANDS`, `score_to_band`, etc.)  
Update to match the scoring criteria sheet.

---

## Setup

### First run (metrics-only)
This confirms the pipeline runs without using any API key:
```

python pipeline_demo.py --json data/01081001-2.json --fps 30 --age-band 9-12 --skill developmental --style mentor
```
### Second run (LLM + LangSmith)
This confirms OpenAI + LangSmith are working:
```
python pipeline_demo.py --json data/01081001-2.json --fps 30 --age-band 9-12 --skill developmental --style mentor --use-llm
```

