# Coach Feedback API — Demo Pipeline (Python)
This repo contains a simple end-to-end **demo pipeline** that:
1. Reads a **Video Annotation JSON** export (from your labeling platform)
2. Computes **metrics** (touch rate, ball control, rhythm, speed, etc.)
3. Converts metrics into **interpretable 0–1** scores using age/skill thresholds
4. Calls the **ChatGPT API** to generate **coach feedback** in a chosen style

It’s designed for quick demos and for iterating on scoring rules later.

## What the JSON input should contain
The script expects your export to include a `labels` list with items like:
* **Positional labels**
    * `"Ball Position"` (frame + x,y)
    * `"Corner 1 of Mat"`, `"Corner 2 of Mat"`, `"Corner 3 of Mat"`, `"Corner 4 of Mat"`
        * Must be in this order:
            1 = top-left, 2 = top-right, 3 = bottom-right, 4 = bottom-left
* **Temporal labels (startFrame/endFrame)**
    * `"Left Ball touch"`, `"Right Ball touch"`
    * `"Toe tap event"`
    * Optional (if your labeling includes them): `"Look down"`, `"Ball out of frame"`, `"Ball out of reach"`
If a label doesn’t exist in your data (e.g., “Look down”), the related metric will just default to a neutral value.

## How the pipeline works
### Step 1 — Parse annotation JSON
* Loads the JSON and separates:
    * positional labels (corners, ball positions)
    * temporal labels (touch events)
### Step 2 — Normalize ball coordinates (homography)
* Uses the 4 mat corners to compute a perspective transform
* Converts ball pixel (x,y) → normalized mat coordinates (u,v) in a unit square (0..1)
* This makes speed/control metrics comparable across clips (even if camera angle changes)
### Step 3 — Compute raw metrics
Examples:
* `touches_per_min`
* `toe_taps_per_min`
* `lr_balance_score` (how evenly left/right touches are distributed)
* `rhythm_score` (steadiness of touch intervals; robust median/MAD method)
* `ball_avg_speed`, `ball_max_speed`, `ball_speed_spikiness`
* `ball_control_score` (penalizes spikiness + out-of-frame/out-of-reach durations if labeled)
### Step 4 — Convert metrics → interpretable scores (0–1)
* Applies age/skill “starter thresholds” (placeholders you can tune later)
* Produces:
    * scores (dict of 0–1 values)
    * strengths (top 0–1 scores)
    * focus (lowest 0–1 scores)
### Step 5 — (Optional) Generate coach feedback via ChatGPT API
* Sends the metrics + scores to the API
* Returns a structured JSON response:
    * praise
    * strengths
    * improvements
    * drill
    * next_goal

## What each important function does
### Parsing
* `load_annotation(path)` → reads JSON file
* `split_labels(labels)` → splits positional vs temporal labels
* `extract_corner_sets(positional)` → finds all corner points (may exist at multiple frames)
* `extract_ball_points(positional)` → ball trajectory points
* `extract_events(temporal)` → event list (name, startFrame, endFrame)
* `estimate_duration_seconds(labels, fps)` → rough clip duration from max frame

### Normalization (mat coordinates)
* `pick_corners_for_frame(corner_sets, frame)` → uses closest corner set for each ball point
* `homography_from_corners(src_corners)` → computes perspective transform
* `normalize_ball_track(ball, corner_sets)` → produces `(frame,x,y,u,v)` track

### Metrics
* `count_events(events)` → counts labels (left touches, right touches, toe taps…)
* `sum_event_durations(events, target, fps)` → total seconds for an event type
* `touch_frames_from_events(events)` → uses midpoint frame for touch timing
* `rhythm_score_from_touches(touch_frames, fps)` → robust rhythm score 0..1
* `speed_stats(track_uv, fps)` → avg/max speed + spikiness
* `compute_metrics(labels, fps)` → returns the full metrics JSON

### Scoring + strengths/focus
* `apply_scores(metrics, age_band, skill)` → adds:
* scores (0..1)
* strengths (up to 3)
* focus (up to 3)

### LLM feedback
* `generate_coach_feedback(metrics, age_band, skill, coach_style)` → calls ChatGPT API and returns structured feedback JSON

## Setup
### 1. Activate your environment
```
source .venv/bin/activate
```
### 2. Install dependencies
```
pip install openai python-dotenv pydantic opencv-python numpy
```
### 3. Add your API key (only for `--use-llm`)
Create a `.env` file in the project root:
```

OPENAI_API_KEY="YOUR_KEY_HERE"
OPENAI_MODEL="gpt-5.2"
```
Make sure `.env` is in `.gitignore`.

## How to run
### Metrics only (no API call)
```

python pipeline_demo.py --json data/01081001-2.json --fps 30 --age-band 9-12 --skill developmental --style mentor
```
### Metrics + coach feedback (API call)
```

python pipeline_demo.py --json data/01081001-2.json --fps 30 --age-band 9-12 --skill developmental --style mentor --use-llm
```
### Try different settings
* `--age-band`: `U8`, `9-12`, `13+`
* `--skill`: `recreational`, `developmental`, `premium_pro`
* `--style`: `cheer`, `mentor`, `performance`



