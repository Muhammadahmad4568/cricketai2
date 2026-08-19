import os
import pandas as pd
import numpy as np

# ============================================================
# CRICKET AI - TECHNIQUE EVALUATION
# ============================================================
# NEW FILE — this did not exist in your uploaded project.
# It's referenced by your brief (Shot #11: 94.44/100, "Elbow:
# GOOD", etc.) and by your own app.py (which already tries to
# load technique_evaluation.csv), but no script generated it —
# shot_analysis.csv only has raw angles and movement, not
# scores or GOOD/BAD ratings.
#
# IMPORTANT: the thresholds below are heuristic placeholders,
# not a trained model. You've already said the real "ideal
# technique" comparison isn't built yet — this is a reasonable
# stand-in that produces real ratings from your real angle data
# so the hackathon demo isn't showing fake numbers. Expect to
# retune ELBOW_GOOD_RANGE / knee-stability thresholds once you
# have labeled footage or the ideal-pose model.
# ============================================================

# Elbow angle band considered "good" for a batting shot.
# Outside this band, the shot is flagged as too closed (bat
# arm collapsed) or too straight (locked/rigid).
ELBOW_GOOD_RANGE = (70, 160)
ELBOW_IDEAL_CENTER = 115

# Knee-angle standard deviation (within a window around the shot)
# above this is treated as an unstable base.
KNEE_STABILITY_STD_THRESHOLD = 12.0
KNEE_WINDOW_FRAMES = 15

# Ball-shot frame-difference bands for timing quality.
TIMING_EXCELLENT_FRAMES = 3
TIMING_GOOD_FRAMES = 7
TIMING_AVERAGE_FRAMES = 15

SCORE_WEIGHTS = {"elbow": 35, "knee": 35, "timing": 30}

OVERALL_THRESHOLDS = [(85, "EXCELLENT"), (70, "GOOD"), (55, "AVERAGE")]


def overall_rating(score):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "N/A"
    for threshold, label in OVERALL_THRESHOLDS:
        if score >= threshold:
            return label
    return "NEEDS IMPROVEMENT"


def evaluate_elbow(angle):
    if angle is None or pd.isna(angle):
        return None, "UNAVAILABLE"

    low, high = ELBOW_GOOD_RANGE

    if angle < low:
        rating = "TOO CLOSED"
        score = max(0.0, 55 - (low - angle) * 1.2)
    elif angle > high:
        rating = "TOO STRAIGHT"
        score = max(0.0, 85 - (angle - high) * 1.5)
    else:
        rating = "GOOD"
        closeness = 1 - (abs(angle - ELBOW_IDEAL_CENTER) / (high - low))
        score = 75 + max(0.0, closeness) * 25

    return round(min(score, 100), 2), rating


def evaluate_knee(avg_angle, window_std):
    if avg_angle is None or pd.isna(avg_angle):
        return None, "UNAVAILABLE"

    if window_std is not None and not pd.isna(window_std):
        if window_std > KNEE_STABILITY_STD_THRESHOLD:
            rating = "UNSTABLE"
            score = max(0.0, 70 - (window_std - KNEE_STABILITY_STD_THRESHOLD) * 2)
        else:
            rating = "GOOD"
            score = 75 + (1 - window_std / KNEE_STABILITY_STD_THRESHOLD) * 25
        return round(min(score, 100), 2), rating

    # Fallback when no smoothed time-series window is available:
    # judge only from the single averaged angle at the shot moment.
    if avg_angle < 90 or avg_angle > 178:
        return 50.0, "UNSTABLE"
    return 75.0, "GOOD"


def evaluate_timing(frame_difference, ball_detected):
    if not ball_detected or frame_difference is None or pd.isna(frame_difference):
        return None, "UNAVAILABLE"

    frame_difference = abs(frame_difference)

    if frame_difference <= TIMING_EXCELLENT_FRAMES:
        return 95.0, "EXCELLENT"
    if frame_difference <= TIMING_GOOD_FRAMES:
        return 80.0, "GOOD"
    if frame_difference <= TIMING_AVERAGE_FRAMES:
        return 60.0, "AVERAGE"
    return 40.0, "NEEDS IMPROVEMENT"


def combined_score(elbow_score, knee_score, timing_score):
    components = {
        "elbow": elbow_score,
        "knee": knee_score,
        "timing": timing_score,
    }

    available = {k: v for k, v in components.items() if v is not None}

    if not available:
        return None

    total_weight = sum(SCORE_WEIGHTS[k] for k in available)
    weighted_sum = sum(SCORE_WEIGHTS[k] * v for k, v in available.items())

    return round(weighted_sum / total_weight, 2)


def build_advice(elbow_rating, knee_rating, timing_rating):
    tips = []

    if elbow_rating == "TOO CLOSED":
        tips.append("Avoid closing the elbow too much through the shot.")
    elif elbow_rating == "TOO STRAIGHT":
        tips.append("Avoid locking the elbow rigidly — allow a natural bend.")

    if knee_rating == "UNSTABLE":
        tips.append("Work on lower-body stability during the shot.")

    if timing_rating == "NEEDS IMPROVEMENT":
        tips.append("Timing relative to the ball was off — focus on watching the ball longer before committing.")
    elif timing_rating == "UNAVAILABLE":
        tips.append("Ball timing could not be verified for this shot (ball detection was unreliable nearby).")

    if not tips:
        tips.append("Solid shot — elbow, knee, and timing all within a good range.")

    return " ".join(tips)


def _knee_window_std(smoothed_df, shot_frame, left_col, right_col, window=KNEE_WINDOW_FRAMES):
    if smoothed_df is None:
        return None

    lo, hi = shot_frame - window, shot_frame + window
    window_df = smoothed_df[(smoothed_df["frame"] >= lo) & (smoothed_df["frame"] <= hi)]

    if window_df.empty:
        return None

    values = []
    for col in [left_col, right_col]:
        if col and col in window_df.columns:
            values.append(pd.to_numeric(window_df[col], errors="coerce"))

    if not values:
        return None

    combined = pd.concat(values)
    return float(combined.std())


def run_technique_evaluation(shot_csv, output_dir, sync_csv=None, smoothed_csv=None):
    """
    Scores each shot in shot_analysis.csv and writes
    technique_evaluation.csv into output_dir.

    Returns the output CSV path.
    """

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "technique_evaluation.csv")

    shots_df = pd.read_csv(shot_csv)

    sync_df = None
    if sync_csv and os.path.exists(sync_csv):
        sync_df = pd.read_csv(sync_csv)

    smoothed_df = None
    knee_cols = (None, None)
    if smoothed_csv and os.path.exists(smoothed_csv):
        smoothed_df = pd.read_csv(smoothed_csv)
        left_col = "left_knee_angle" if "left_knee_angle" in smoothed_df.columns else None
        right_col = "right_knee_angle" if "right_knee_angle" in smoothed_df.columns else None
        knee_cols = (left_col, right_col)

    rows = []

    for _, shot in shots_df.iterrows():

        shot_number = shot["shot_number"]
        shot_frame = shot["frame"]

        elbow_angle = shot.get("average_elbow_angle")
        knee_angle = shot.get("average_knee_angle")
        movement_score = shot.get("movement_score")

        elbow_score, elbow_rating = evaluate_elbow(elbow_angle)

        knee_std = _knee_window_std(smoothed_df, shot_frame, *knee_cols)
        knee_score, knee_rating = evaluate_knee(knee_angle, knee_std)

        frame_difference = None
        ball_detected = False
        ball_confidence = None

        if sync_df is not None:
            match = sync_df[sync_df["shot_number"] == shot_number]
            if not match.empty:
                match_row = match.iloc[0]
                frame_difference = match_row.get("frame_difference")
                ball_detected = bool(match_row.get("ball_detected", False))
                ball_confidence = match_row.get("ball_confidence")

        timing_score, timing_rating = evaluate_timing(frame_difference, ball_detected)

        score = combined_score(elbow_score, knee_score, timing_score)
        overall = overall_rating(score)
        advice = build_advice(elbow_rating, knee_rating, timing_rating)

        rows.append({
            "shot_number": shot_number,
            "frame": shot_frame,
            "time_seconds": shot.get("time_seconds"),
            "technique_score": score,
            "overall": overall,
            "elbow_angle": elbow_angle,
            "elbow": elbow_rating,
            "knee_angle": knee_angle,
            "knee": knee_rating,
            "timing": timing_rating,
            "movement_score": movement_score,
            "ball_confidence": ball_confidence,
            "frame_difference": frame_difference,
            "advice": advice,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)

    return output_csv


# ============================================================
# CLI MODE
# ============================================================

if __name__ == "__main__":

    print("Evaluating technique...")

    path = run_technique_evaluation(
        shot_csv="shot_analysis.csv",
        output_dir=".",
        sync_csv="ball_shot_sync.csv" if os.path.exists("ball_shot_sync.csv") else None,
        smoothed_csv="smoothed_pose_data.csv" if os.path.exists("smoothed_pose_data.csv") else None,
    )

    df = pd.read_csv(path)

    print()
    print("========================================")
    print("   TECHNIQUE EVALUATION COMPLETE")
    print("========================================")
    print(f"Output file: {path}")
    print(f"Shots evaluated: {len(df)}")
    valid_scores = df["technique_score"].dropna()
    if not valid_scores.empty:
        print(f"Average technique score: {valid_scores.mean():.2f}/100")
