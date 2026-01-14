#!/usr/bin/env python3
"""
pipeline_demo.py
----------------
Demo pipeline for U-PRO "coach feedback" prototypes.

Supports TWO input formats:
1) Annotation JSON (from the Video Annotator tool)
2) Session-output JSON (from the analytics pipeline / backend)

What it does:
- Parse input JSON
- Compute / extract a compact set of metrics
- Convert key metrics into 0–100 "scores" (where available)
- Apply the U-PRO scoring rubric (bands: 1–20, 21–40, 41–60, 61–80, 81–100)
- Optionally call an LLM to generate coach-style feedback
- Optional LangSmith tracing (so your supervisor can see prompts/inputs)

Run:
  # Metrics only
  python pipeline_demo.py --json data/01081001-2.json --fps 30 --age-band 9-12 --skill developmental --style mentor

  # Metrics + coach feedback (LLM)
  python pipeline_demo.py --json data/01081001-2.json --fps 30 --age-band 9-12 --skill developmental --style mentor --use-llm

Env (.env recommended):
  OPENAI_API_KEY=...
  OPENAI_MODEL=gpt-5.2

  # LangSmith (optional)
  LANGSMITH_API_KEY=...
  LANGSMITH_TRACING=true
  LANGSMITH_PROJECT=coach-feedback-api
  LANGCHAIN_HIDE_INPUTS=false
  LANGCHAIN_HIDE_OUTPUTS=false
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional: load .env automatically when running this script
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# =============================================================================
# Scoring rubric (0–100)
# -----------------------------------------------------------------------------
# Based on the "U-Pro Soccer - Metrics and Scoring Criteria" sheet:
# Each metric uses a 5-band interpretation:
#   1–20, 21–40, 41–60, 61–80, 81–100
# NOTE: These are *descriptions*, not formulas. We apply them AFTER we have
# a score in 0–100 from the analytics pipeline (or a proxy score in demos).
# =============================================================================

RUBRIC_BANDS: List[Tuple[int, int, str]] = [
    (1, 20, "Needs improvement"),
    (21, 40, "Below average"),
    (41, 60, "Average"),
    (61, 80, "Above average"),
    (81, 100, "Excellent"),
]

# Compact descriptions (kept short for LangSmith readability)
UPRO_RUBRIC: Dict[str, Dict[str, str]] = {
    "head_up": {
        "1-20": "Player does not attempt to keep head up.",
        "21-40": "Head up rarely; mainly looks down.",
        "41-60": "Head up sometimes; still frequent head-down.",
        "61-80": "Head up often; good awareness with occasional head-down.",
        "81-100": "Head up consistently; strong awareness and decision-making.",
    },
    "ball_control": {
        "1-20": "Touches are uncontrolled; ball is often lost.",
        "21-40": "Some control but frequent mistakes; ball often escapes.",
        "41-60": "Fair control; occasional ball loss; improving touch quality.",
        "61-80": "Good control; ball mostly close; few mistakes.",
        "81-100": "Excellent control; consistent, close touches and mastery.",
    },
    "technique_coordination": {
        "1-20": "Struggles with technique; coordination is poor.",
        "21-40": "Basic technique but inconsistent; coordination needs work.",
        "41-60": "Acceptable technique; coordination improving.",
        "61-80": "Good technique; coordinated movements with minor issues.",
        "81-100": "Excellent technique and coordination; fluid, consistent form.",
    },
    "speed_agility": {
        "1-20": "Very slow; difficulty changing tempo/direction.",
        "21-40": "Slow and inconsistent; limited agility.",
        "41-60": "Moderate speed; some agility; needs consistency.",
        "61-80": "Good speed and responsiveness; agile with small improvements.",
        "81-100": "Excellent speed/agility; quick reactions and smooth control.",
    },
    "balance_power": {
        "1-20": "Very unstable; limited power/efficiency.",
        "21-40": "Unstable at times; power output inconsistent.",
        "41-60": "Reasonable balance; average power/efficiency.",
        "61-80": "Good balance; solid power/efficiency.",
        "81-100": "Excellent balance and controlled power; efficient movement.",
    },
    "effort_endurance": {
        "1-20": "Low effort; poor endurance; stops frequently.",
        "21-40": "Effort inconsistent; endurance needs improvement.",
        "41-60": "Average effort; can complete drill with fatigue.",
        "61-80": "Good sustained effort; strong endurance for age/level.",
        "81-100": "Excellent effort and endurance; consistent intensity.",
    },
}

# =============================================================================
# Mat-only drills (safety constraints)
# =============================================================================

MAT_SAFE_DRILLS = {
    "U8": [
        "Toe taps (count to 10, rest, repeat)",
        "Inside-inside touches (tiny touches, keep the ball close)",
        "Sole stop + move (stop ball with sole, push gently to the other foot)",
    ],
    "9-12": [
        "Metronome touches: side-to-side at a steady count (1-2-1-2) for 30s x 3",
        "Toe taps ladder: 15s easy, 15s steady, 15s fast x 2 rounds",
        "Inside touches square: 4 corners of the mat, light touches, keep rhythm",
    ],
    "13+": [
        "Tempo control: 20s steady cadence + 10s faster cadence, repeat x 4",
        "Weak-foot emphasis: 45s mostly weak-foot touches, keep ball centered",
        "Rhythm reset: if rhythm breaks, slow down 3 touches then build back up",
    ],
}

STYLE_HINTS = {
    "cheer": "Very encouraging, simple, short sentences. One clear next step.",
    "mentor": "Supportive and practical: 2 strengths, 2 improvements, 1 drill, 1 measurable goal.",
    "performance": "Direct and metrics-focused. Mention numbers lightly. Clear priorities and targets.",
}


# =============================================================================
# Utilities
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def score_band(score_0_100: float) -> Tuple[str, str]:
    """Return ('61-80', 'Above average') etc."""
    s = int(round(clamp(score_0_100, 0, 100)))
    if s == 0:
        # allow 0 as "no attempt / no data"
        return "1-20", "Needs improvement"
    for lo, hi, label in RUBRIC_BANDS:
        if lo <= s <= hi:
            key = f"{lo}-{hi}"
            return key, label
    return "81-100", "Excellent"


def rubric_lookup(metric_key: str, score_0_100: float) -> Dict[str, Any]:
    band_key, band_label = score_band(score_0_100)
    desc = UPRO_RUBRIC.get(metric_key, {}).get(band_key, "")
    return {
        "score_0_100": round(clamp(score_0_100, 0, 100), 2),
        "band": band_key,
        "band_label": band_label,
        "criteria": desc,
    }


def get_nested(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# =============================================================================
# Input detection & parsing
# =============================================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def identify_input_format(data: Dict[str, Any]) -> str:
    """
    Returns:
      - "annotation" for Video Annotator JSON
      - "session_output" for analytics/session output JSON
    """
    # Annotation JSON typically has 'labels' or 'events' and 'video' metadata
    if any(k in data for k in ["labels", "events", "video", "temporal_labels", "positional_labels"]):
        return "annotation"
    # Session output typically has drill_id / session_id / profile_id, or nested analysis blocks
    if any(k in data for k in ["session_id", "profile_id", "drill_id", "ball_control_analysis", "head_up_analysis"]):
        return "session_output"
    # Fallback: try best guess by structure
    return "annotation"


# =============================================================================
# Annotation JSON -> demo metrics
# =============================================================================

def _extract_temporal_events(annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Support multiple schemas: "events" or "labels" etc.
    if isinstance(annotation.get("events"), list):
        return annotation["events"]
    if isinstance(annotation.get("labels"), list):
        # Some tools store temporal labels in labels[] with timestamps
        return annotation["labels"]
    return []


def _count_events(events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in events:
        name = e.get("label") or e.get("name") or e.get("type") or "unknown"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _estimate_duration_seconds(events: List[Dict[str, Any]], fps: float) -> float:
    """
    Best-effort: if events contain frame indices or timestamps, estimate duration.
    Otherwise, fallback to touches / fps heuristic.
    """
    # try timestamps in seconds
    t0 = None
    t1 = None
    for e in events:
        t = e.get("t") or e.get("time") or e.get("timestamp")
        if isinstance(t, (int, float)):
            t0 = t if t0 is None else min(t0, t)
            t1 = t if t1 is None else max(t1, t)
    if t0 is not None and t1 is not None and t1 >= t0:
        return float(t1 - t0)

    # try frames
    f0 = None
    f1 = None
    for e in events:
        fr = e.get("frame") or e.get("start_frame") or e.get("end_frame")
        if isinstance(fr, int):
            f0 = fr if f0 is None else min(f0, fr)
            f1 = fr if f1 is None else max(f1, fr)
    if f0 is not None and f1 is not None and f1 >= f0 and fps:
        return float(f1 - f0) / float(fps)

    # fallback heuristic
    return 0.0


def compute_metrics_from_annotations(annotation: Dict[str, Any], fps: float) -> Dict[str, Any]:
    events = _extract_temporal_events(annotation)
    counts = _count_events(events)

    left = counts.get("Left Ball touch", 0) + counts.get("Left ball touch", 0) + counts.get("left_touch", 0)
    right = counts.get("Right Ball touch", 0) + counts.get("Right ball touch", 0) + counts.get("right_touch", 0)
    toe = counts.get("Toe tap event", 0) + counts.get("Toe taps", 0) + counts.get("toe_tap", 0)

    touches = left + right
    duration_s = _estimate_duration_seconds(events, fps)

    # If duration is unknown, approximate from frames in filename metadata or touches rate.
    if duration_s <= 0 and fps > 0:
        # crude heuristic: assume ~2.2 touches/sec if we have touch events
        duration_s = safe_div(touches, 2.2)

    touches_per_min = safe_div(touches, duration_s) * 60.0
    toe_taps_per_min = safe_div(toe, duration_s) * 60.0

    # Demo "scores" (0–1) derived from simple heuristics
    lr_balance_score = 1.0 - (abs(left - right) / touches) if touches else 0.0
    lr_balance_score = clamp(lr_balance_score, 0.0, 1.0)

    # heads_up_score: from Look down counts if present, else assume good (demo)
    look_down = counts.get("Look down", 0) + counts.get("look_down", 0)
    # penalty: each look-down counts as ~0.15 in this toy demo
    heads_up_score = clamp(1.0 - 0.15 * look_down, 0.0, 1.0)

    # rhythm_score: use variability proxy if we have enough events; else 0.5
    rhythm_score = 0.5 if touches >= 10 else 0.0

    # ball_control_score: proxy using spikiness if present (not available => use balance)
    ball_control_score = clamp(0.5 + 0.5 * lr_balance_score, 0.0, 1.0)

    # Output
    metrics = {
        "duration_s_est": duration_s,
        "touches": touches,
        "left_touches": left,
        "right_touches": right,
        "touches_per_min": touches_per_min,
        "toe_taps": toe,
        "toe_taps_per_min": toe_taps_per_min,

        # 0–1 "demo scores"
        "lr_balance_score": lr_balance_score,
        "heads_up_score": heads_up_score,
        "rhythm_score": rhythm_score,
        "ball_control_score": ball_control_score,

        "event_counts": counts,
    }

    # Scorecard using rubric (0–100)
    scorecard = {
        "head_up": rubric_lookup("head_up", heads_up_score * 100),
        "ball_control": rubric_lookup("ball_control", ball_control_score * 100),
        "technique_coordination": rubric_lookup("technique_coordination", ((lr_balance_score + rhythm_score) / 2) * 100),
        "speed_agility": rubric_lookup("speed_agility", clamp(touches_per_min / 180.0, 0, 1) * 100),
    }

    return {"metrics": metrics, "scorecard": scorecard}


# =============================================================================
# Session-output JSON -> extracted metrics + rubric
# =============================================================================

@dataclass
class SessionExtract:
    session_id: Optional[int] = None
    profile_id: Optional[int] = None
    drill_id: Optional[int] = None
    duration_s: Optional[float] = None

    # Ball control sub-metrics
    total_touches: Optional[int] = None
    left_touches: Optional[int] = None
    right_touches: Optional[int] = None

    ground_touch_percentage: Optional[float] = None
    ball_presence_percentage: Optional[float] = None
    ball_out_of_reach_count: Optional[int] = None
    avg_recovery_time_s: Optional[float] = None

    # Scores (0–100 if available)
    head_up_score_0_100: Optional[float] = None
    ball_control_score_0_100: Optional[float] = None
    technique_score_0_100: Optional[float] = None
    coordination_score_0_100: Optional[float] = None
    speed_score_0_100: Optional[float] = None
    agility_score_0_100: Optional[float] = None
    balance_score_0_100: Optional[float] = None
    power_score_0_100: Optional[float] = None
    endurance_score_0_100: Optional[float] = None


def compute_metrics_from_session_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact subset of metrics (per supervisor feedback: avoid too many totals/percentages).
    We keep:
      - Touches (left/right)
      - Ground touch percentage
      - Ball presence percentage
      - Out-of-reach count + average recovery time
      - Summary scores (0–100) if provided by upstream analytics
    """
    s = SessionExtract()

    # IDs
    s.session_id = data.get("session_id")
    s.profile_id = data.get("profile_id")
    s.drill_id = data.get("drill_id")

    # Duration
    s.duration_s = to_float(
        data.get("duration_s")
        or get_nested(data, ["ball_control_analysis", "duration_s"])
        or get_nested(data, ["ball_control_analysis", "duration_s_est"]),
        default=0.0,
    )

    # Ball control analysis block (supports a few possible key styles)
    bc = data.get("ball_control_analysis") or data.get("ballControlAnalysis") or {}
    if not isinstance(bc, dict):
        bc = {}

    s.total_touches = int(bc.get("total_touches") or bc.get("touches") or data.get("total_touches") or 0)
    s.left_touches = int(bc.get("left_touches") or data.get("left_touches") or 0)
    s.right_touches = int(bc.get("right_touches") or data.get("right_touches") or 0)

    # Ground touch %
    s.ground_touch_percentage = to_float(
        bc.get("ground_touch_percentage")
        or get_nested(bc, ["ground_touches", "ground_touch_percentage"])
        or get_nested(data, ["ground_touches", "ground_touch_percentage"]),
        default=0.0,
    )

    # Ball presence %
    s.ball_presence_percentage = to_float(
        bc.get("ball_presence_percentage")
        or get_nested(bc, ["ball_presence_detection", "ball_presence_percentage"])
        or get_nested(data, ["ball_presence_detection", "ball_presence_percentage"]),
        default=0.0,
    )

    # Out of reach count + recovery time
    s.ball_out_of_reach_count = int(
        bc.get("ball_out_of_reach_count")
        or get_nested(bc, ["ball_out_of_reach_analysis", "out_of_reach_count"])
        or get_nested(data, ["ball_out_of_reach_analysis", "out_of_reach_count"])
        or 0
    )
    s.avg_recovery_time_s = to_float(
        bc.get("avg_recovery_time_s")
        or get_nested(bc, ["ball_out_of_reach_analysis", "avg_recovery_time_s"])
        or get_nested(data, ["ball_out_of_reach_analysis", "avg_recovery_time_s"]),
        default=0.0,
    )

    # Summary score block (0–100) – upstream may provide these already
    # We support:
    #  - data["summary_scores_0_100"]
    #  - data["scores"]
    #  - nested: head_up_detailed_score.overall_score, etc.
    score_block = data.get("summary_scores_0_100") or data.get("scores") or {}
    if not isinstance(score_block, dict):
        score_block = {}

    s.head_up_score_0_100 = to_float(
        score_block.get("head_up")
        or get_nested(data, ["head_up_detailed_score", "overall_score"])
        or get_nested(data, ["head_up_score", "overall_score"]),
        default=0.0,
    )
    s.ball_control_score_0_100 = to_float(score_block.get("ball_control"), default=0.0)
    s.technique_score_0_100 = to_float(score_block.get("technique"), default=0.0)
    s.coordination_score_0_100 = to_float(score_block.get("coordination"), default=0.0)
    s.speed_score_0_100 = to_float(score_block.get("speed"), default=0.0)
    s.agility_score_0_100 = to_float(score_block.get("agility"), default=0.0)
    s.balance_score_0_100 = to_float(score_block.get("balance"), default=0.0)
    s.power_score_0_100 = to_float(score_block.get("power"), default=0.0)
    s.endurance_score_0_100 = to_float(score_block.get("endurance"), default=0.0)

    extracted = {
        "session_id": s.session_id,
        "profile_id": s.profile_id,
        "drill_id": s.drill_id,
        "duration_s": s.duration_s,

        # Ball control (compact)
        "touches": {
            "total_touches": s.total_touches,
            "left_touches": s.left_touches,
            "right_touches": s.right_touches,
        },
        "ground_touch_percentage": s.ground_touch_percentage,
        "ball_presence_percentage": s.ball_presence_percentage,
        "ball_out_of_reach": {
            "count": s.ball_out_of_reach_count,
            "avg_recovery_time_s": s.avg_recovery_time_s,
        },

        # Scores (if available)
        "scores_0_100": {
            "head_up": s.head_up_score_0_100,
            "ball_control": s.ball_control_score_0_100,
            "technique": s.technique_score_0_100,
            "coordination": s.coordination_score_0_100,
            "speed": s.speed_score_0_100,
            "agility": s.agility_score_0_100,
            "balance": s.balance_score_0_100,
            "power": s.power_score_0_100,
            "endurance": s.endurance_score_0_100,
        },
    }

    # Scorecard via rubric (only include metrics that have a non-zero value)
    scorecard: Dict[str, Any] = {}

    if s.head_up_score_0_100 and s.head_up_score_0_100 > 0:
        scorecard["head_up"] = rubric_lookup("head_up", s.head_up_score_0_100)
    if s.ball_control_score_0_100 and s.ball_control_score_0_100 > 0:
        scorecard["ball_control"] = rubric_lookup("ball_control", s.ball_control_score_0_100)

    # Combine technique + coordination if both exist; else use whichever exists
    tc = 0.0
    tc_n = 0
    for v in [s.technique_score_0_100, s.coordination_score_0_100]:
        if v and v > 0:
            tc += float(v)
            tc_n += 1
    if tc_n:
        scorecard["technique_coordination"] = rubric_lookup("technique_coordination", tc / tc_n)

    # Speed/agility
    sa = 0.0
    sa_n = 0
    for v in [s.speed_score_0_100, s.agility_score_0_100]:
        if v and v > 0:
            sa += float(v)
            sa_n += 1
    if sa_n:
        scorecard["speed_agility"] = rubric_lookup("speed_agility", sa / sa_n)

    # Balance/power
    bp = 0.0
    bp_n = 0
    for v in [s.balance_score_0_100, s.power_score_0_100]:
        if v and v > 0:
            bp += float(v)
            bp_n += 1
    if bp_n:
        scorecard["balance_power"] = rubric_lookup("balance_power", bp / bp_n)

    # Effort/endurance (if present)
    if s.endurance_score_0_100 and s.endurance_score_0_100 > 0:
        scorecard["effort_endurance"] = rubric_lookup("effort_endurance", s.endurance_score_0_100)

    return {"metrics": extracted, "scorecard": scorecard}


# =============================================================================
# LLM coach feedback (OpenAI + optional LangSmith)
# =============================================================================

def truthy_env(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def get_openai_client():
    """
    Returns an OpenAI client, optionally wrapped for LangSmith tracing.
    """
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file.")

    client = OpenAI(api_key=api_key)

    # Optional: LangSmith trace wrapper (lets you see prompts/inputs/outputs)
    # Requires: pip install langsmith
    # And env vars:
    #   LANGSMITH_API_KEY, LANGSMITH_TRACING=true
    if truthy_env("LANGSMITH_TRACING") and os.getenv("LANGSMITH_API_KEY"):
        try:
            from langsmith.wrappers import wrap_openai  # type: ignore
            client = wrap_openai(client)
        except Exception:
            # If wrapper not available, still run without tracing
            pass

    return client


def select_mat_drill(age_band: str) -> str:
    key = age_band
    if age_band.strip().lower() in {"u8", "under 8", "8"}:
        key = "U8"
    elif age_band.strip() in {"9-12", "9–12"}:
        key = "9-12"
    elif age_band.strip() in {"13+", "13", "14+", "15+"}:
        key = "13+"
    pool = MAT_SAFE_DRILLS.get(key, MAT_SAFE_DRILLS["9-12"])
    return random.choice(pool)


def build_prompt(
    metrics_block: Dict[str, Any],
    scorecard: Dict[str, Any],
    age_band: str,
    skill: str,
    style: str,
) -> Tuple[str, str]:
    system = (
        "You are a youth soccer training assistant. "
        "You must ONLY suggest drills that can be done on a 4x4 feet training mat. "
        "Do NOT suggest jogging, passing, shooting, or long-space dribbling. "
        "Keep it safe and age-appropriate. "
        "Output MUST be valid JSON with keys: praise, strengths, improvements, drill, next_goal."
    )

    style_hint = STYLE_HINTS.get(style, STYLE_HINTS["mentor"])

    user = {
        "context": {
            "age_band": age_band,
            "skill_level": skill,
            "style": style,
            "style_hint": style_hint,
        },
        "metrics": metrics_block,
        "scorecard": scorecard,
        "constraints": {
            "mat_only": True,
            "space": "4x4 feet",
            "no_kicking_passing_shooting": True,
        },
    }

    return system, json.dumps(user, ensure_ascii=False)


def call_llm_for_feedback(
    metrics_block: Dict[str, Any],
    scorecard: Dict[str, Any],
    age_band: str,
    skill: str,
    style: str,
) -> Dict[str, Any]:
    client = get_openai_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    system, user = build_prompt(metrics_block, scorecard, age_band, skill, style)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
    )

    text = resp.choices[0].message.content or ""
    # Best effort: parse JSON from model output
    try:
        return json.loads(text)
    except Exception:
        return {"raw_text": text}


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to input JSON (annotation or session-output).")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for annotation JSON duration estimation.")
    parser.add_argument("--age-band", default="9-12", help="e.g., U8, 9-12, 13+")
    parser.add_argument("--skill", default="developmental", help="recreational | developmental | premium/pro")
    parser.add_argument("--style", default="mentor", help="cheer | mentor | performance")
    parser.add_argument("--use-llm", action="store_true", help="Call the LLM to generate coach feedback.")
    parser.add_argument("--input-type", default="auto", help="auto | annotation | session_output")
    args = parser.parse_args()

    data = load_json(args.json)

    input_type = args.input_type
    if input_type == "auto":
        input_type = identify_input_format(data)

    if input_type == "annotation":
        out = compute_metrics_from_annotations(data, fps=args.fps)
    elif input_type == "session_output":
        out = compute_metrics_from_session_output(data)
    else:
        raise ValueError(f"Unknown input-type: {input_type}")

    metrics_block = out["metrics"]
    scorecard = out.get("scorecard", {})

    print("\n=== METRICS ===")
    print(json.dumps(metrics_block, indent=2, ensure_ascii=False))

    if scorecard:
        print("\n=== SCORECARD (Rubric) ===")
        print(json.dumps(scorecard, indent=2, ensure_ascii=False))

    if args.use_llm:
        print("\n=== COACH FEEDBACK (Structured JSON) ===")
        feedback = call_llm_for_feedback(metrics_block, scorecard, args.age_band, args.skill, args.style)
        # If the LLM didn't provide a drill, inject a safe mat drill
        if isinstance(feedback, dict) and "drill" not in feedback:
            feedback["drill"] = select_mat_drill(args.age_band)
        print(json.dumps(feedback, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
