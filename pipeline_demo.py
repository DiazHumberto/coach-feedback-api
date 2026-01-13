#!/usr/bin/env python3
"""
pipeline_demo.py

Demo pipeline for U-Pro / coach-feedback prototype.

Supports TWO input formats:

1) Annotation export JSON (from the intern labeling tool)
   - top-level key: "labels"
   - positional labels: {type:"positional", name, frame, x, y}
   - temporal labels:   {type:"temporal", name, startFrame, endFrame}

2) Session-output JSON (from the internal analysis engine)
   - top-level keys like: "ball_control_analysis", "balance_and_stability_analysis",
     "form_and_technique_analysis", "speed_agility_rhythm_analysis", etc.

The pipeline outputs:
- METRICS (normalized, JSON-safe)
- optionally COACH FEEDBACK (structured JSON) using OpenAI API

NOTE: This script is intentionally "prototype-friendly":
- deterministic metrics/scoring in Python
- LLM used only for phrasing + drill suggestions (mat-safe)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

# Optional deps (annotation mode uses cv2 for homography; session mode does not require it)
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # noqa

import math

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # noqa


# --------------------------
# JSON loading (robust)
# --------------------------

_NUMPY_FLOAT_RE = re.compile(r"np\.float\d*\(([^)]*)\)")
_NUMPY_INT_RE = re.compile(r"np\.int\d*\(([^)]*)\)")
_ARRAY_RE = re.compile(r"array\((\[[\s\S]*?\])\)")

def load_json_or_python_repr(path: str) -> Dict[str, Any]:
    """
    Load either:
    - valid JSON
    - or a Python dict/list repr that may include np.float64(...) and array([...])

    This is helpful because some "session outputs" are pasted/serialized from Python prints.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # 1) Try strict JSON first
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Try to sanitize common numpy prints and then literal_eval
    sanitized = raw
    sanitized = _NUMPY_FLOAT_RE.sub(r"\1", sanitized)
    sanitized = _NUMPY_INT_RE.sub(r"\1", sanitized)
    sanitized = _ARRAY_RE.sub(r"\1", sanitized)

    # Some prints include "np.float64(100.0)" inside dicts with single quotes.
    # literal_eval can parse that after sanitization.
    try:
        obj = ast.literal_eval(sanitized)
    except Exception as e:
        raise ValueError(
            "Could not parse file as JSON or as a sanitized Python dict repr. "
            "If this came from a printout, try exporting as JSON."
        ) from e

    if not isinstance(obj, dict):
        raise ValueError("Expected top-level dict for session/annotation file.")
    return obj


def to_json_safe(x: Any) -> Any:
    """Convert numpy-like values, sets, tuples into JSON-safe Python types."""
    # numpy scalars often have .item()
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            pass

    if isinstance(x, dict):
        return {str(k): to_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json_safe(v) for v in x]
    if isinstance(x, set):
        return [to_json_safe(v) for v in sorted(list(x))]
    return x


# --------------------------
# Input type detection
# --------------------------

def detect_input_type(data: Dict[str, Any]) -> str:
    # Annotation export format
    if isinstance(data.get("labels"), list):
        return "annotation"
    # Session output format
    if any(k in data for k in ("ball_control_analysis", "summary_scores_0_100", "form_and_technique_analysis")):
        return "session"
    return "unknown"


# --------------------------
# Annotation mode: metrics
# --------------------------

@dataclass
class CornerSet:
    tl: Tuple[float, float]
    tr: Tuple[float, float]
    br: Tuple[float, float]
    bl: Tuple[float, float]


def _extract_corners(labels: List[Dict[str, Any]]) -> Optional[CornerSet]:
    """Extract corners from positional labels (Corner 1..4 of Mat)."""
    pts: Dict[str, Tuple[float, float]] = {}
    for l in labels:
        if l.get("type") != "positional":
            continue
        name = l.get("name")
        if name in ("Corner 1 of Mat", "Corner 2 of Mat", "Corner 3 of Mat", "Corner 4 of Mat"):
            pts[name] = (float(l["x"]), float(l["y"]))

    needed = ["Corner 1 of Mat", "Corner 2 of Mat", "Corner 3 of Mat", "Corner 4 of Mat"]
    if not all(n in pts for n in needed):
        return None

    # User confirmed: 1=TL, 2=TR, 3=BR, 4=BL
    return CornerSet(
        tl=pts["Corner 1 of Mat"],
        tr=pts["Corner 2 of Mat"],
        br=pts["Corner 3 of Mat"],
        bl=pts["Corner 4 of Mat"],
    )


def _extract_ball_track(labels: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
    track = []
    for l in labels:
        if l.get("type") == "positional" and l.get("name") == "Ball Position":
            track.append((int(l["frame"]), float(l["x"]), float(l["y"])))
    track.sort(key=lambda t: t[0])
    return track


def _extract_events(labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = []
    for l in labels:
        if l.get("type") == "temporal":
            events.append(l)
    return events


def _estimate_duration_s(labels: List[Dict[str, Any]], fps: float) -> float:
    max_frame = 0
    for l in labels:
        if l.get("type") == "positional":
            max_frame = max(max_frame, int(l.get("frame", 0)))
        elif l.get("type") == "temporal":
            max_frame = max(max_frame, int(l.get("endFrame", 0)))
    # Include frame 0 -> frame max inclusive => (max_frame+1)/fps
    return float(max_frame + 1) / float(fps)


def _count_event_name(events: List[Dict[str, Any]], name: str) -> int:
    return sum(1 for e in events if e.get("name") == name)


def _lr_balance_score(left: int, right: int) -> float:
    total = left + right
    if total <= 0:
        return 0.0
    return 1.0 - abs(left - right) / total


def _touch_mid_frames(events: List[Dict[str, Any]]) -> List[int]:
    frames = []
    for e in events:
        if e.get("name") in ("Left Ball touch", "Right Ball touch"):
            s = int(e.get("startFrame", 0))
            t = int(e.get("endFrame", s))
            frames.append((s + t) // 2)
    frames.sort()
    return frames


def _rhythm_score_from_touches(touch_frames: List[int], fps: float) -> float:
    """
    Simple rhythm score:
    - compute inter-touch intervals (seconds)
    - score = 1 - (std / (mean + eps)), clamped to [0,1]
    """
    if len(touch_frames) < 6:
        return 0.0
    intervals = []
    for a, b in zip(touch_frames, touch_frames[1:]):
        dt = (b - a) / fps
        if 0.05 <= dt <= 2.5:
            intervals.append(dt)
    if len(intervals) < 5:
        return 0.0
    mean = sum(intervals) / len(intervals)
    var = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    std = math.sqrt(var)
    score = 1.0 - (std / (mean + 1e-6))
    return float(max(0.0, min(1.0, score)))


def _homography_pixel_to_unit(corners: CornerSet) -> Optional[Any]:
    """Return 3x3 transform H mapping pixel -> unit square coords."""
    if cv2 is None:
        return None

    import numpy as np  # local import so script still loads without numpy errors elsewhere

    src = np.array([corners.tl, corners.tr, corners.br, corners.bl], dtype=np.float32)
    dst = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def _normalize_track_to_unit(track: List[Tuple[int, float, float]], H: Any) -> List[Tuple[int, float, float]]:
    """Apply homography to track; returns (frame, u, v)."""
    import numpy as np

    if not track:
        return []
    pts = np.array([[x, y] for _, x, y in track], dtype=np.float32).reshape(-1, 1, 2)
    uv = cv2.perspectiveTransform(pts, H).reshape(-1, 2)  # type: ignore
    out = []
    for (frame, _, _), (u, v) in zip(track, uv):
        out.append((frame, float(u), float(v)))
    return out


def _speed_stats(track_uv: List[Tuple[int, float, float]], fps: float) -> Dict[str, float]:
    if len(track_uv) < 3:
        return {"ball_avg_speed": 0.0, "ball_max_speed": 0.0, "ball_speed_spikiness": 0.0}

    speeds = []
    for (f1, u1, v1), (f2, u2, v2) in zip(track_uv, track_uv[1:]):
        dt = (f2 - f1) / fps
        if dt <= 0:
            continue
        dist = math.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
        speeds.append(dist / dt)

    if not speeds:
        return {"ball_avg_speed": 0.0, "ball_max_speed": 0.0, "ball_speed_spikiness": 0.0}

    avg = sum(speeds) / len(speeds)
    mx = max(speeds)
    var = sum((s - avg) ** 2 for s in speeds) / len(speeds)
    std = math.sqrt(var)
    spiky = std / (avg + 1e-6)
    return {"ball_avg_speed": float(avg), "ball_max_speed": float(mx), "ball_speed_spikiness": float(spiky)}


def compute_metrics_from_annotation(data: Dict[str, Any], fps: float) -> Dict[str, Any]:
    labels = data.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError("Annotation input must have a list under 'labels'.")

    corners = _extract_corners(labels)
    track = _extract_ball_track(labels)
    events = _extract_events(labels)

    duration_s = _estimate_duration_s(labels, fps)
    left = _count_event_name(events, "Left Ball touch")
    right = _count_event_name(events, "Right Ball touch")
    toe = _count_event_name(events, "Toe tap event")
    touches = left + right

    touches_per_min = (touches / duration_s) * 60.0 if duration_s > 0 else 0.0
    toe_per_min = (toe / duration_s) * 60.0 if duration_s > 0 else 0.0

    lr_balance = _lr_balance_score(left, right)

    touch_mid = _touch_mid_frames(events)
    rhythm = _rhythm_score_from_touches(touch_mid, fps)

    # Heads-up in annotation mode depends on labels you may not have.
    # If you do not label head-down events, we keep it at 1.0 (neutral/high).
    heads_up = 1.0

    ball_stats = {"ball_avg_speed": 0.0, "ball_max_speed": 0.0, "ball_speed_spikiness": 0.0}
    if corners and track and cv2 is not None:
        H = _homography_pixel_to_unit(corners)
        if H is not None:
            track_uv = _normalize_track_to_unit(track, H)
            ball_stats = _speed_stats(track_uv, fps)

    # ball_control_score: simple prototype combination (lower spikiness => better)
    spiky = float(ball_stats["ball_speed_spikiness"])
    ball_control_score = float(max(0.0, min(1.0, 1.0 - (spiky / 1.2))))  # heuristic scale

    metrics = {
        "input_type": "annotation",
        "duration_s_est": duration_s,
        "touches": touches,
        "left_touches": left,
        "right_touches": right,
        "touches_per_min": touches_per_min,
        "toe_taps": toe,
        "toe_taps_per_min": toe_per_min,
        "lr_balance_score": lr_balance,
        "heads_up_score": heads_up,
        "rhythm_score": rhythm,
        **ball_stats,
        "ball_control_score": ball_control_score,
        "event_counts": {
            "Left Ball touch": left,
            "Right Ball touch": right,
            "Toe tap event": toe,
        },
    }
    return metrics


# --------------------------
# Session-output mode: adapt
# --------------------------

def _get_nested(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def compute_metrics_from_session_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt a session-output dict into the same "headline metrics" schema used by the pipeline.
    """
    # Duration
    fps = _get_nested(data, ["video", "fps"], None)
    duration_s = (
        _get_nested(data, ["video", "duration_s"], None)
        or _get_nested(data, ["ball_control_analysis", "duration_s"], None)
    )
    if duration_s is None and fps is not None:
        nframes = _get_nested(data, ["video", "num_frames"], 0)
        duration_s = float(nframes) / float(fps) if nframes else None
    duration_s = float(duration_s) if duration_s is not None else 0.0

    # Ball control
    bc = data.get("ball_control_analysis", {}) if isinstance(data.get("ball_control_analysis"), dict) else {}
    touches = int(bc.get("total_touches", 0) or 0)
    left = int(bc.get("left_touches", 0) or 0)
    right = int(bc.get("right_touches", 0) or 0)

    # Cadence (touches per minute)
    cadence_tpm = bc.get("cadence_tpm", None)
    if cadence_tpm is None and duration_s > 0:
        cadence_tpm = (touches / duration_s) * 60.0
    cadence_tpm = float(cadence_tpm or 0.0)

    lr_balance = _lr_balance_score(left, right)

    # Heads up
    # Prefer explicit summary_scores_0_100.heads_up if present
    heads_up_0_100 = (
        _get_nested(data, ["summary_scores_0_100", "heads_up"], None)
        or _get_nested(data, ["form_and_technique_analysis", "head_movement_score_0_100"], None)
        or _get_nested(data, ["form_and_technique_analysis", "head_up_detail_score", "overall_score_0_100"], None)
    )
    heads_up_score = float(heads_up_0_100) / 100.0 if heads_up_0_100 is not None else 0.0

    # Balance
    balance_0_100 = (
        _get_nested(data, ["summary_scores_0_100", "balance"], None)
        or _get_nested(data, ["balance_and_stability_analysis", "balance_score_0_100"], None)
    )
    balance_score = float(balance_0_100) / 100.0 if balance_0_100 is not None else 0.0

    # Ball control score (0..1)
    bc_0_100 = _get_nested(data, ["summary_scores_0_100", "ball_control"], None)
    ball_control_score = float(bc_0_100) / 100.0 if bc_0_100 is not None else 0.0

    # Rhythm: use "cadence_score_0_100" (proxy) if present
    cadence_score_0_100 = _get_nested(data, ["speed_agility_rhythm_analysis", "cadence_score_0_100"], None)
    rhythm_score = float(cadence_score_0_100) / 100.0 if cadence_score_0_100 is not None else 0.0

    # Extras
    look_down_pct = _get_nested(data, ["form_and_technique_analysis", "look_down_percentage"], None)
    look_down_pct = float(look_down_pct) if look_down_pct is not None else None

    ball_presence_pct = _get_nested(data, ["ball_control_analysis", "submetrics", "ball_presence_detection", "ball_presence_percentage"], None)
    ball_presence_pct = float(ball_presence_pct) if ball_presence_pct is not None else None

    out_reach_pct = _get_nested(data, ["ball_control_analysis", "submetrics", "ball_out_of_reach_analysis", "out_of_reach_percentage"], None)
    out_reach_pct = float(out_reach_pct) if out_reach_pct is not None else None

    metrics = {
        "input_type": "session",
        "duration_s_est": duration_s,
        "touches": touches,
        "left_touches": left,
        "right_touches": right,
        "touches_per_min": cadence_tpm,
        "lr_balance_score": lr_balance,
        "heads_up_score": heads_up_score,
        "balance_score": balance_score,
        "rhythm_score": rhythm_score,
        "ball_control_score": ball_control_score,
        "look_down_percentage": look_down_pct,
        "ball_presence_percentage": ball_presence_pct,
        "ball_out_of_reach_percentage": out_reach_pct,
        "raw_summary_scores_0_100": to_json_safe(_get_nested(data, ["summary_scores_0_100"], {})),
    }
    return metrics


# --------------------------
# Scoring interpretation
# --------------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def interpret_strengths_and_focus(metrics: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Simple rule: take a shortlist of key scores, pick top 2 and bottom 2.
    (This keeps LLM prompt stable and deterministic.)
    """
    key_scores = {
        "Ball control": float(metrics.get("ball_control_score", 0.0) or 0.0),
        "Heads up": float(metrics.get("heads_up_score", 0.0) or 0.0),
        "Rhythm": float(metrics.get("rhythm_score", 0.0) or 0.0),
        "Left/right balance": float(metrics.get("lr_balance_score", 0.0) or 0.0),
    }
    # Include balance if available
    if "balance_score" in metrics and metrics["balance_score"] is not None:
        key_scores["Balance"] = float(metrics.get("balance_score", 0.0) or 0.0)

    ranked = sorted(key_scores.items(), key=lambda kv: kv[1], reverse=True)
    strengths = [k for k, _ in ranked[:2]]
    focus = [k for k, _ in ranked[-2:]]
    return strengths, focus


# --------------------------
# LLM Coach feedback
# --------------------------

MAT_SAFE_DRILLS = {
    "U8": [
        "Toe taps (count to 10, rest, repeat)",
        "Inside-inside touches (tiny touches, keep the ball close)",
        "Sole stop + move (stop ball with sole, push gently to the other foot)"
    ],
    "9-12": [
        "Metronome touches: side-to-side at a steady count (1-2-1-2) for 30s x 3",
        "Toe taps ladder: 15s easy, 15s steady, 15s fast x 2 rounds",
        "Inside touches square: 4 corners of the mat, light touches, keep rhythm"
    ],
    "13+": [
        "Tempo control: 20s steady cadence + 10s faster cadence, repeat x 4",
        "Weak-foot emphasis: 45s mostly weak-foot touches, keep ball centered",
        "Rhythm reset: if rhythm breaks, slow down 3 touches then build back up"
    ]
}

STYLE_HINTS = {
    "cheer": "Very encouraging, simple, short sentences. Emphasize effort and one clear next step.",
    "mentor": "Supportive and practical. 2 strengths, 2 improvements, 1 drill, 1 measurable goal.",
    "performance": "Direct and metrics-focused. Mention numbers lightly. Clear priorities and targets."
}

def generate_coach_feedback_llm(
    metrics: Dict[str, Any],
    age_band: str,
    skill: str,
    style: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls OpenAI Chat Completions API and requests STRICT JSON output.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your environment or .env file.")

    # Load dotenv if available and user uses .env
    if load_dotenv is not None:
        load_dotenv(override=False)

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package not installed. Run: pip install openai") from e

    client = OpenAI(api_key=api_key)

    strengths, focus = interpret_strengths_and_focus(metrics)
    drill_options = MAT_SAFE_DRILLS.get(age_band, MAT_SAFE_DRILLS["9-12"])

    system = (
        "You are a youth soccer training assistant. "
        "You must ONLY suggest drills that can be done on a 4x4 feet training mat. "
        "Do NOT suggest jogging, passing, shooting, or long-space dribbling. "
        "Keep it safe and age-appropriate."
    )

    user = {
        "task": "Generate coach feedback as STRICT JSON.",
        "player_profile": {"age_band": age_band, "skill_level": skill},
        "coach_style": style,
        "style_hint": STYLE_HINTS.get(style, STYLE_HINTS["mentor"]),
        "constraints": {
            "environment": "4x4_ft_mat_only",
            "allowed_drill_examples": drill_options,
            "must_not_include": ["jogging", "running", "passing", "shooting", "kicking drills", "large-space dribbling"],
        },
        "metrics": metrics,
        "computed_summary": {"strengths": strengths, "focus": focus},
        "required_json_schema": {
            "praise": "string",
            "strengths": ["string", "string"],
            "improvements": ["string", "string"],
            "drill": "string (mat-safe)",
            "next_goal": "string (measurable, next session)"
        }
    }

    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    # Prefer response_format JSON if supported
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(text)
    except Exception:
        # Fallback: ask without response_format and parse as JSON
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "Return ONLY valid JSON for the following:\n" + json.dumps(user)},
            ],
            temperature=0.4,
        )
        text = resp.choices[0].message.content or "{}"
        # try best-effort parse
        try:
            return json.loads(text)
        except Exception:
            # Attempt to extract JSON object from text
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
            raise RuntimeError("LLM did not return valid JSON.")


# --------------------------
# CLI
# --------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Coach feedback pipeline demo (annotation or session-output input).")
    parser.add_argument("--json", required=True, help="Path to input file (annotation JSON or session-output JSON).")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS (annotation mode). Ignored in session mode if video.fps exists.")
    parser.add_argument("--age-band", dest="age_band", default="9-12", choices=["U8", "9-12", "13+"], help="Player age band.")
    parser.add_argument("--skill", default="developmental", choices=["recreational", "developmental", "premium_pro"], help="Skill tier.")
    parser.add_argument("--style", default="mentor", choices=["cheer", "mentor", "performance"], help="Coach feedback style.")
    parser.add_argument("--use-llm", action="store_true", help="Generate coach feedback using OpenAI API.")
    parser.add_argument("--input-type", default="auto", choices=["auto", "annotation", "session"], help="Force input type or auto-detect.")
    parser.add_argument("--out", default=None, help="Optional path to save combined output JSON.")
    args = parser.parse_args()

    # Load .env if available
    if load_dotenv is not None:
        load_dotenv(override=False)

    data = load_json_or_python_repr(args.json)

    input_type = args.input_type
    if input_type == "auto":
        input_type = detect_input_type(data)
        if input_type == "unknown":
            raise ValueError("Could not auto-detect input type. Use --input-type annotation|session.")

    if input_type == "annotation":
        metrics = compute_metrics_from_annotation(data, fps=args.fps)
    elif input_type == "session":
        metrics = compute_metrics_from_session_output(data)
    else:
        raise ValueError("Unsupported input type.")

    # Add derived summaries
    strengths, focus = interpret_strengths_and_focus(metrics)
    metrics["strengths"] = strengths
    metrics["focus"] = focus

    print("\n=== METRICS ===")
    print(json.dumps(to_json_safe(metrics), indent=2))

    output: Dict[str, Any] = {"metrics": to_json_safe(metrics)}

    if args.use_llm:
        feedback = generate_coach_feedback_llm(
            metrics=to_json_safe(metrics),
            age_band=args.age_band,
            skill=args.skill,
            style=args.style,
        )
        print("\n=== COACH FEEDBACK (Structured JSON) ===")
        print(json.dumps(to_json_safe(feedback), indent=2))
        output["coach_feedback"] = to_json_safe(feedback)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(to_json_safe(output), f, indent=2)
        print(f"\nSaved output to: {args.out}")


if __name__ == "__main__":
    main()
