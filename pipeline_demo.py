import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from dotenv import load_dotenv

FPS_DEFAULT = 30.0

CORNERS_ORDER = [
    "Corner 1 of Mat",  # top-left
    "Corner 2 of Mat",  # top-right
    "Corner 3 of Mat",  # bottom-right
    "Corner 4 of Mat",  # bottom-left
]

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
    corner_sets: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for l in positional:
        name = l.get("name")
        if name in CORNERS_ORDER:
            frame = int(l["frame"])
            corner_sets.setdefault(frame, {})[name] = (float(l["x"]), float(l["y"]))
    return corner_sets

def extract_ball_points(positional: List[dict]) -> List[Tuple[int, float, float]]:
    ball = []
    for l in positional:
        if l.get("name") == "Ball Position":
            ball.append((int(l["frame"]), float(l["x"]), float(l["y"])))
    ball.sort(key=lambda t: t[0])
    return ball

def extract_events(temporal: List[dict]) -> List[Tuple[str, int, int]]:
    events = []
    for l in temporal:
        events.append((l["name"], int(l["startFrame"]), int(l["endFrame"])))
    return events

def estimate_duration_seconds(labels: List[dict], fps: float) -> float:
    max_frame = 0
    for l in labels:
        if l.get("type") == "positional":
            max_frame = max(max_frame, int(l.get("frame", 0)))
        elif l.get("type") == "temporal":
            max_frame = max(max_frame, int(l.get("endFrame", 0)))
    return max_frame / fps if fps > 0 else 0.0

# ---------------------------
# Normalization (homography)
# ---------------------------

def pick_corners_for_frame(corner_sets: Dict[int, Dict[str, Tuple[float, float]]], frame: int) -> Optional[np.ndarray]:
    if not corner_sets:
        return None
    closest_frame = min(corner_sets.keys(), key=lambda f: abs(f - frame))
    corners_dict = corner_sets[closest_frame]
    if not all(c in corners_dict for c in CORNERS_ORDER):
        return None
    return np.array([corners_dict[c] for c in CORNERS_ORDER], dtype=np.float32)  # 4x2

def homography_from_corners(src_corners: np.ndarray) -> np.ndarray:
    # Map pixel corners -> unit square (u,v)
    dst = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_corners, dst)

def normalize_ball_track(ball: List[Tuple[int, float, float]], corner_sets: Dict[int, Dict[str, Tuple[float, float]]]):
    # returns list of (frame, x, y, u, v)
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
# Metrics (simple but useful)
# ---------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

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

def touch_times(events: List[Tuple[str, int, int]]) -> List[int]:
    # treat startFrame as the event moment
    frames = [s for name, s, _ in events if name in ("Left Ball touch", "Right Ball touch")]
    frames.sort()
    return frames

def rhythm_score_from_touches(touch_frames: List[int], fps: float) -> float:
    # Lower variation in intervals => better rhythm
    if len(touch_frames) < 4:
        return 0.0
    intervals = np.diff(np.array(touch_frames)) / fps
    mean = float(np.mean(intervals))
    std = float(np.std(intervals))
    if mean <= 1e-6:
        return 0.0
    cv = std / mean  # coefficient of variation
    # Map: cv 0.0 -> 1.0 (great), cv 0.6+ -> ~0 (rough)
    return clamp01(1.0 - (cv / 0.6))

def speed_stats(track_uv: List[Tuple[int, float, float, float, float]], fps: float):
    if len(track_uv) < 2:
        return {"avg_speed": 0.0, "max_speed": 0.0, "speed_spikiness": 0.0}
    speeds = []
    for i in range(1, len(track_uv)):
        f0, _, _, u0, v0 = track_uv[i-1]
        f1, _, _, u1, v1 = track_uv[i]
        dt = (f1 - f0) / fps
        if dt <= 0:
            continue
        dist = ((u1-u0)**2 + (v1-v0)**2) ** 0.5
        speeds.append(dist / dt)
    if not speeds:
        return {"avg_speed": 0.0, "max_speed": 0.0, "speed_spikiness": 0.0}
    speeds = np.array(speeds, dtype=float)
    avg = float(np.mean(speeds))
    mx = float(np.max(speeds))
    spiky = float(np.std(speeds) / (avg + 1e-6))  # higher = less controlled
    return {"avg_speed": avg, "max_speed": mx, "speed_spikiness": spiky}

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
    lr_balance = 1.0 - (abs(left - right) / (touches + 1e-6))  # 1.0 best

    look_down_s = sum_event_durations(events, "Look down", fps)
    look_down_ratio = look_down_s / duration_s
    heads_up_score = clamp01(1.0 - look_down_ratio)

    # Normalize ball
    corner_sets = extract_corner_sets(positional)
    ball = extract_ball_points(positional)
    track_uv = normalize_ball_track(ball, corner_sets)
    sp = speed_stats(track_uv, fps)

    # Simple control score: penalize spikiness + out-of-frame/out-of-reach
    out_frame_s = sum_event_durations(events, "Ball out of frame", fps)
    out_reach_s = sum_event_durations(events, "Ball out of reach", fps)
    penalty = (out_frame_s + out_reach_s) / duration_s
    control_score = clamp01(1.0 - (sp["speed_spikiness"] / 2.0) - penalty)

    rhythm = rhythm_score_from_touches(touch_times(events), fps)

    return {
        "duration_s_est": duration_s,
        "touches": touches,
        "left_touches": left,
        "right_touches": right,
        "touches_per_min": touches_per_min,
        "toe_taps": toe_taps,
        "toe_taps_per_min": toe_taps_per_min,
        "lr_balance_score": float(clamp01(lr_balance)),
        "heads_up_score": float(heads_up_score),
        "rhythm_score": float(rhythm),
        "ball_avg_speed": sp["avg_speed"],
        "ball_max_speed": sp["max_speed"],
        "ball_speed_spikiness": sp["speed_spikiness"],
        "ball_control_score": float(control_score),
        "event_counts": counts,
    }

# ---------------------------
# LLM Feedback (Structured Output)
# ---------------------------

def generate_coach_feedback(metrics: dict, age_band: str, skill_level: str, coach_style: str):
    # Lazy import so the script still works without OpenAI
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
        "Here are the computed performance metrics (0–1 scores are higher=better):\n"
        + json.dumps(metrics, indent=2)
        + "\nReturn feedback for the player based on these metrics."
    )

    # Structured output using the Python SDK + Pydantic
    # (This matches OpenAI's Structured Outputs guide usage of client.responses.parse.)
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
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

    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))

    if args.use_llm:
        print("\n=== COACH FEEDBACK (Structured JSON) ===")
        feedback = generate_coach_feedback(metrics, args.age_band, args.skill, args.style)
        print(json.dumps(feedback, indent=2))

if __name__ == "__main__":
    main()