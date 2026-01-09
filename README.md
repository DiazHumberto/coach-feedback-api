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
Temporal labels (startFrame/endFrame)
"Left Ball touch", "Right Ball touch"
"Toe tap event"
Optional (if your labeling includes them): "Look down", "Ball out of frame", "Ball out of reach"
If a label doesn’t exist in your data (e.g., “Look down”), the related metric will just default to a neutral value.

How the pipeline works (high level)
Step 1 — Parse annotation JSON
Loads the JSON and separates:
positional labels (corners, ball positions)
temporal labels (touch events)
Step 2 — Normalize ball coordinates (homography)
Uses the 4 mat corners to compute a perspective transform
Converts ball pixel (x,y) → normalized mat coordinates (u,v) in a unit square (0..1)
This makes speed/control metrics comparable across clips (even if camera angle changes)
Step 3 — Compute raw metrics
Examples:
touches_per_min
toe_taps_per_min
lr_balance_score (how evenly left/right touches are distributed)
rhythm_score (steadiness of touch intervals; robust median/MAD method)
ball_avg_speed, ball_max_speed, ball_speed_spikiness
ball_control_score (penalizes spikiness + out-of-frame/out-of-reach durations if labeled)
Step 4 — Convert metrics → interpretable scores (0–1)
Applies age/skill “starter thresholds” (placeholders you can tune later)
Produces:
scores (dict of 0–1 values)
strengths (top 0–1 scores)
focus (lowest 0–1 scores)
Step 5 — (Optional) Generate coach feedback via ChatGPT API
Sends the metrics + scores to the API
Returns a structured JSON response:
praise
strengths
improvements
drill
next_goal