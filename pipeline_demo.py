import argparse
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from dotenv import load_dotenv

FPS_DEFAULT = 30.0

# Mat corner labels (fixed order you confirmed)
CORNERS_ORDER = [
    "Corner 1 of Mat",  # top-left
    "Corner 2 of Mat",  # top-right
    "Corner 3 of Mat",  # bottom-right
    "Corner 4 of Mat",  # bottom-left
]

# ---------------------------
# Helpers
# ---------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def linear_score(x: float, low: float, high: float, higher_is_better: bool = True) -> float:
    """
    Map x to 0..1 between low..high.
    If higher_is_better=False, lower values are better.
    """
    if high <= low:
        base = 1.0 if x >= high else 0.0
        return clamp01(base if higher_is_better else (1.0 - base))

    t = (x - low) / (high - low)
    t = clamp01(t)
    return t if higher_is_better else (1.0 - t)

# ---------------------------
# Parsing
# ---------------------------

def load_annotation(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_labels(labels: List[dict]):
    positional = [l for l in labels if l.get("type") == "positional"]
    temporal = [l for l in labels if l.get("type") == "temporal"]
    return positional, temporal

def extract_corner_sets(positional: List[dict]) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Returns: {frame: {"Corner 1 of Mat": (x,y), ...}}
    """
    corner_sets: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for l in positional:
        name = l.get("name")
        if name in CORNERS_ORDER:
            frame = int(l["frame"])
            corner_sets.setdefault(frame, {})[name] = (float(l["x"]), float(l["y"]))
    return corner_sets

def extract_ball_points(positional: List[dict]) -> List[Tuple[int, float, float]]:
    """
    Returns list of (frame, x, y) for "Ball Position".
    """
    ball = []
    for l in positional:
        if l.get("name") == "Ball Position":
            ball.append((int(l["frame"]), float(l["x"]), float(l["y"])))
    ball.sort(key=lambda t: t[0])
    return ball

def extract_events(temporal: List[dict]) -> List[Tuple[str, int, int]]:
    """
    Returns list of (name, startFrame, endFrame).
    """
    events = []
    for l in temporal:
        events.append((l["name"], int(l["startFrame"]), int(l["endFrame"])))
    return events

def estimate_duration_seconds(labels: List[dict], fps: float) -> float:
    """
    Estimate clip duration from max frame encountered in labels.
    """
    max_frame = 0
    for l in labels:
        if l.get("type") == "positional":
            max_frame = max(max_frame, int(l.get("frame", 0)))
        elif l.get("type") == "temporal":
            max_frame = max(max_frame, int(l.get("endFrame", 0)))
    return (max_frame / fps) if fps > 0 else 0.0

# ---------------------------
# Normalization (homography)
# ---------------------------

def pick_corners_for_frame(
    corner_sets: Dict[int, Dict[str, Tuple[float, float]]],
    frame: int
) -> Optional[np.ndarray]:
    """
    Pick the corner set whose frame is closest to 'frame'.
    """
    if not corner_sets:
        return None

    closest_frame = min(corner_sets.keys(), key=lambda f: abs(f - frame))
    corners_dict = corner_sets[closest_frame]

    if not all(c in corners_dict for c in CORNERS_ORDER):
        return None

    return np.array([corners_dict[c] for c in CORNERS_ORDER], dtype=np.float32)

def homography_from_corners(src_corners: np.ndarray) -> np.ndarray:
    """
    src_corners: 4x2 in TL,TR,BR,BL (pixels)
    Map to unit square (u,v): (0,0),(1,0),(1,1),(0,1)
    """
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_corners, dst)

def normalize_ball_track(
    ball: List[Tuple[int, float, float]],
    corner_sets: Dict[int, Dict[str, Tuple[float, float]]]
) -> List[Tuple[int, float, float, float, float]]:
    """
    Returns list of (frame, x, y, u, v) where u,v are normalized mat coords.
    """
    out = []
    for frame, x, y in ball:
        src = pick_corners_for_frame(corner_sets, frame)
        if src is None:
            continue
        H = homography_from_corners(src)
        pt = np.array([[[x, y]]], dtype=np.float32)
        uv = cv2.perspectiveTransform(pt, H)[0, 0]
        out.append((frame, x, y, float(uv[0]), float(uv[1])))
    return out

# ---------------------------
# Event/metric utilities
# ---------------------------

def count_events(events: List[Tuple[str, int, int]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, _, _ in events:
        counts[name] = counts.get(name, 0) + 1
    return counts

def sum_event_durations(events: List[Tuple[str, int, int]], target_name: str, fps: float) -> float:
    total = 0.0
    for name, s, e in events:
        if name == target_name:
            total += max(0, (e - s)) / fps
    return total

def touch_frames_from_events(events: List[Tuple[str, int, int]]) -> List[int]:
    """
    Use the midpoint of the temporal segment as the touch "time".
    This is more robust than using startFrame only.
    """
    frames = []
    for name, s, e in events:
        if name in ("Left Ball touch", "Right Ball touch"):
            frames.append(int((s + e) / 2))
    frames.sort()
    return frames

def rhythm_score_from_touches(touch_frames: List[int], fps: float) -> float:
    """
    Robust rhythm score based on touch interval consistency.
    - Uses median/MAD (stable vs mean/std)
    - Filters out tiny intervals and long pauses
    Returns 0..1 (higher = steadier rhythm)
    """
    if len(touch_frames) < 6:
        return 0.0

    times = np.array(touch_frames, dtype=float) / fps
    intervals = np.diff(times)

    # Filter unrealistic intervals:
    intervals = intervals[(intervals > 0.05) & (intervals < 1.0)]
    if len(intervals) < 5:
        return 0.0

    med = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals - med))) + 1e-6

    robust_cv = (1.4826 * mad) / (med + 1e-6)  # approx std/mean

    # Map to score: robust_cv 0 -> 1, robust_cv 0.5+ -> ~0
    return clamp01(1.0 - (robust_cv / 0.5))

def speed_stats(track_uv: List[Tuple[int, float, float, float, float]], fps: float):
    """
    Compute avg/max speed and a 'spikiness' (std/mean) on normalized mat coords.
    """
    if len(track_uv) < 2:
        return {"avg_speed": 0.0, "max_speed": 0.0, "speed_spikiness": 0.0}

    speeds = []
    for i in range(1, len(track_uv)):
        f0, _, _, u0, v0 = track_uv[i - 1]
        f1, _, _, u1, v1 = track_uv[i]
        dt = (f1 - f0) / fps
        if dt <= 0:
            continue
        dist = ((u1 - u0) ** 2 + (v1 - v0) ** 2) ** 0.5
        speeds.append(dist / dt)

    if not speeds:
        return {"avg_speed": 0.0, "max_speed": 0.0, "speed_spikiness": 0.0}

    speeds = np.array(speeds, dtype=float)
    avg = float(np.mean(speeds))
    mx = float(np.max(speeds))
    spiky = float(np.std(speeds) / (avg + 1e-6))
    return {"avg_speed": avg, "max_speed": mx, "speed_spikiness": spiky}

# ---------------------------
# Core metrics
# ---------------------------

def compute_metrics(labels: List[dict], fps: float):
    positional, temporal = split_labels(labels)
    events = extract_events(temporal)
    counts = count_events(events)

    duration_s = estimate_duration_seconds(labels, fps)
    duration_s = max(duration_s, 1e-6)

    left = counts.get("Left Ball touch", 0)
    right = counts.get("Right Ball touch", 0)
    touches = left + right

    toe_taps = counts.get("Toe tap event", 0)

    touches_per_min = (touches / duration_s) * 60.0
    toe_taps_per_min = (toe_taps / duration_s) * 60.0

    lr_balance_score = 1.0 - (abs(left - right) / (touches + 1e-6))  # 1.0 best

    # Heads up score (only works if "Look down" events exist)
    look_down_s = sum_event_durations(events, "Look down", fps)
    look_down_ratio = look_down_s / duration_s
    heads_up_score = clamp01(1.0 - look_down_ratio)

    # Ball normalization + speed features
    corner_sets = extract_corner_sets(positional)
    ball = extract_ball_points(positional)
    track_uv = normalize_ball_track(ball, corner_sets)
    sp = speed_stats(track_uv, fps)

    # Rhythm from touch timestamps
    tframes = touch_frames_from_events(events)
    rhythm_score = rhythm_score_from_touches(tframes, fps)

    # Control score: penalize spikiness + out-of-frame/out-of-reach (if present)
    out_frame_s = sum_event_durations(events, "Ball out of frame", fps)
    out_reach_s = sum_event_durations(events, "Ball out of reach", fps)
    penalty = (out_frame_s + out_reach_s) / duration_s
    control_score = clamp01(1.0 - (sp["speed_spikiness"] / 2.0) - penalty)

    return {
        "duration_s_est": duration_s,
        "touches": touches,
        "left_touches": left,
        "right_touches": right,
        "touches_per_min": touches_per_min,
        "toe_taps": toe_taps,
        "toe_taps_per_min": toe_taps_per_min,
        "lr_balance_score": float(clamp01(lr_balance_score)),
        "heads_up_score": float(heads_up_score),
        "rhythm_score": float(rhythm_score),
        "ball_avg_speed": sp["avg_speed"],
        "ball_max_speed": sp["max_speed"],
        "ball_speed_spikiness": sp["speed_spikiness"],
        "ball_control_score": float(control_score),
        "event_counts": counts,
    }

# ---------------------------
# Point 3: Threshold-based scores (interpretable 0..1)
# ---------------------------

# Starter thresholds (placeholders; tune later with real dataset stats)
THRESHOLDS = {
    "U8": {
        "recreational":   {"touches_per_min": (20, 80),  "ball_control_score": (0.40, 0.75), "rhythm_score": (0.10, 0.55)},
        "developmental":  {"touches_per_min": (30, 100), "ball_control_score": (0.45, 0.80), "rhythm_score": (0.15, 0.60)},
        "premium_pro":    {"touches_per_min": (40, 120), "ball_control_score": (0.50, 0.85), "rhythm_score": (0.20, 0.65)},
    },
    "9-12": {
        "recreational":   {"touches_per_min": (40, 110), "ball_control_score": (0.45, 0.80), "rhythm_score": (0.15, 0.65)},
        "developmental":  {"touches_per_min": (60, 140), "ball_control_score": (0.50, 0.85), "rhythm_score": (0.20, 0.70)},
        "premium_pro":    {"touches_per_min": (70, 160), "ball_control_score": (0.55, 0.90), "rhythm_score": (0.25, 0.75)},
    },
    "13+": {
        "recreational":   {"touches_per_min": (50, 120), "ball_control_score": (0.50, 0.85), "rhythm_score": (0.20, 0.70)},
        "developmental":  {"touches_per_min": (70, 150), "ball_control_score": (0.55, 0.90), "rhythm_score": (0.25, 0.75)},
        "premium_pro":    {"touches_per_min": (80, 170), "ball_control_score": (0.60, 0.92), "rhythm_score": (0.30, 0.80)},
    },
}

PRETTY = {
    "touch_rate": "Touch rate",
    "rhythm": "Rhythm",
    "ball_control": "Ball control",
    "touch_smoothness": "Smooth touches",
    "lr_balance": "Left/right balance",
    "heads_up": "Heads up",
}

def apply_scores(metrics: dict, age_band: str, skill: str) -> dict:
    cfg = THRESHOLDS.get(age_band, {}).get(skill)
    if cfg is None:
        cfg = {"touches_per_min": (50, 140), "ball_control_score": (0.50, 0.85), "rhythm_score": (0.20, 0.70)}

    t_low, t_high = cfg["touches_per_min"]
    c_low, c_high = cfg["ball_control_score"]
    r_low, r_high = cfg["rhythm_score"]

    scores = {}
    scores["touch_rate"] = linear_score(metrics["touches_per_min"], t_low, t_high, higher_is_better=True)
    scores["ball_control"] = linear_score(metrics["ball_control_score"], c_low, c_high, higher_is_better=True)
    scores["rhythm"] = linear_score(metrics["rhythm_score"], r_low, r_high, higher_is_better=True)

    # already 0..1
    scores["lr_balance"] = clamp01(metrics.get("lr_balance_score", 0.0))
    scores["heads_up"] = clamp01(metrics.get("heads_up_score", 0.0))

    # spikiness: lower is better (rough heuristic range)
    scores["touch_smoothness"] = linear_score(
        metrics.get("ball_speed_spikiness", 1.0),
        low=0.40,
        high=1.20,
        higher_is_better=False,
    )

    metrics["scores"] = scores

    strengths = [PRETTY[k] for k, v in scores.items() if v >= 0.70]
    focus = [PRETTY[k] for k, v in scores.items() if v <= 0.40]

    metrics["strengths"] = strengths[:3]
    metrics["focus"] = focus[:3]
    return metrics

# ---------------------------
# LLM Feedback (Structured Output)
# ---------------------------

def generate_coach_feedback(metrics: dict, age_band: str, skill_level: str, coach_style: str):
    """
    Requires:
      - OPENAI_API_KEY in environment or .env
      - optional OPENAI_MODEL in environment or .env
    """
    from pydantic import BaseModel, Field
    from openai import OpenAI
    import os

    client = OpenAI()

    class CoachFeedback(BaseModel):
        praise: str = Field(..., description="1 supportive sentence.")
        strengths: List[str] = Field(..., description="2 short bullets.")
        improvements: List[str] = Field(..., description="1-2 short bullets.")
        drill: str = Field(..., description="One simple drill suggestion.")
        next_goal: str = Field(..., description="One measurable goal for next session.")

    system = (
        "You are a youth soccer coach. Create constructive, age-appropriate feedback.\n"
        "Rules:\n"
        "- Be encouraging, never harsh.\n"
        "- Do not mention gender or body/appearance.\n"
        "- Keep language appropriate for kids/teens.\n"
        "- Keep each item short and actionable.\n"
        f"Coach style: {coach_style}\n"
        f"Player age band: {age_band}\n"
        f"Skill level: {skill_level}\n"
    )

    user = (
        "Here are the computed metrics and derived scores (0–1 is higher=better):\n"
        + json.dumps(metrics, indent=2)
        + "\nUse strengths/focus if present, and tailor drill difficulty to skill level."
    )

    model_name = os.getenv("OPENAI_MODEL", "gpt-5.2")

    # Structured output using parse()
    response = client.responses.parse(
        model=model_name,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=CoachFeedback,
    )

    return response.output_parsed.model_dump()

# ---------------------------
# CLI
# ---------------------------

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to annotation JSON export")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--age-band", required=True, choices=["U8", "9-12", "13+"])
    ap.add_argument("--skill", required=True, choices=["recreational", "developmental", "premium_pro"])
    ap.add_argument("--style", required=True, choices=["cheer", "mentor", "performance"])
    ap.add_argument("--use-llm", action="store_true")
    args = ap.parse_args()

    data = load_annotation(args.json)
    metrics = compute_metrics(data["labels"], fps=args.fps)

    # Point 3 additions (scores + strengths/focus)
    metrics = apply_scores(metrics, args.age_band, args.skill)

    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))

    if args.use_llm:
        print("\n=== COACH FEEDBACK (Structured JSON) ===")
        feedback = generate_coach_feedback(metrics, args.age_band, args.skill, args.style)
        print(json.dumps(feedback, indent=2))

if __name__ == "__main__":
    main()
