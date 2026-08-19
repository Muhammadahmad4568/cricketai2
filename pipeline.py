import os
import uuid
import traceback

from ultralytics import YOLO

from combined_detection import run_combined_detection
from pose_smoothing import run_pose_smoothing
from shot_analysis import run_shot_analysis
from ball_shot_sync import run_ball_shot_sync
from technique_evaluation import run_technique_evaluation
from overlay_final_video import run_overlay_final_video

# ============================================================
# CRICKET AI - PIPELINE ORCHESTRATOR
# ============================================================
# CHANGED (SPEED FIX): previously this called pose_analysis.py,
# ball_detection.py, and final_analysis_video.py as three separate
# stages — each of which re-ran YOLO inference over the whole
# video, for 4 total inference passes. It now calls
# combined_detection.py (1 pass, both models) + overlay_final_video.py
# (fast, no model inference) instead — same detections, same
# thresholds, far fewer passes. Both models are also now loaded
# ONCE per run and reused, instead of each stage reloading its own
# copy of the same .pt file.
#
# pose_analysis.py, ball_detection.py, and final_analysis_video.py
# are UNCHANGED and still work standalone/CLI if you want to run a
# single stage manually — this file just doesn't call them anymore.
# ============================================================

STAGE_LABELS = [
    "Loading video",
    "Detecting player pose + cricket ball",
    "Smoothing pose data",
    "Detecting shots",
    "Synchronizing ball + shot",
    "Evaluating technique",
    "Generating final annotated video",
    "Preparing AI Coach",
]


def run_analysis(video_path, work_dir, pose_model_path, ball_model_path,
                  status_callback=None):
    """
    Runs the full CricketAI pipeline on video_path.

    status_callback(stage_index, stage_label, sub_progress_fraction_or_None)
    is called as each stage starts/progresses, for driving a UI progress bar.

    Returns a results dict — same shape as before, plus
    results["video_web_compatible"] (bool) telling app.py whether the
    final video was successfully re-encoded for inline browser playback.
    """

    os.makedirs(work_dir, exist_ok=True)

    results = {
        "video_path": video_path,
        "pose_csv": None,
        "smoothed_csv": None,
        "shot_csv": None,
        "ball_csv": None,
        "sync_csv": None,
        "technique_csv": None,
        "final_video": None,
        "video_web_compatible": None,
        "shot_images": {},
        "errors": {},
        "summary": None,
    }

    def report(stage_index, fraction=None):
        if status_callback:
            status_callback(stage_index, STAGE_LABELS[stage_index], fraction)

    def frame_progress(stage_index):
        def _cb(current, total):
            report(stage_index, min(current / total, 1.0) if total else None)
        return _cb

    report(0, 1.0)

    # ---------------- LOAD MODELS ONCE ----------------
    try:
        pose_model = YOLO(pose_model_path)
        ball_model = YOLO(ball_model_path)
    except Exception as e:
        results["errors"]["model_loading"] = f"{e}\n{traceback.format_exc(limit=2)}"
        report(7, 1.0)
        return results

    # ---------------- COMBINED POSE + BALL DETECTION ----------------
    annotated_video = None
    try:
        report(1, 0.0)
        combined_result = run_combined_detection(
            video_path, work_dir, pose_model, ball_model,
            progress_callback=frame_progress(1),
        )
        results["pose_csv"] = combined_result["pose_csv"]
        results["ball_csv"] = combined_result["ball_csv"]
        annotated_video = combined_result["annotated_video"]
    except Exception as e:
        results["errors"]["combined_detection"] = f"{e}\n{traceback.format_exc(limit=2)}"

    # ---------------- POSE SMOOTHING ----------------
    if results["pose_csv"]:
        try:
            report(2, 0.5)
            results["smoothed_csv"] = run_pose_smoothing(results["pose_csv"], work_dir)
        except Exception as e:
            results["errors"]["pose_smoothing"] = str(e)

    # ---------------- SHOT ANALYSIS ----------------
    if results["smoothed_csv"]:
        try:
            report(3, 0.5)
            results["shot_csv"] = run_shot_analysis(results["smoothed_csv"], work_dir)
        except Exception as e:
            results["errors"]["shot_analysis"] = str(e)

    # ---------------- BALL / SHOT SYNC ----------------
    if results["shot_csv"] and results["ball_csv"]:
        try:
            report(4, 0.5)
            results["sync_csv"] = run_ball_shot_sync(results["shot_csv"], results["ball_csv"], work_dir)
        except Exception as e:
            results["errors"]["ball_shot_sync"] = str(e)
    elif not results["ball_csv"]:
        results["errors"]["ball_shot_sync"] = "Skipped: ball detection did not produce data."
    elif not results["shot_csv"]:
        results["errors"]["ball_shot_sync"] = "Skipped: shot analysis did not produce data."

    # ---------------- TECHNIQUE EVALUATION ----------------
    if results["shot_csv"]:
        try:
            report(5, 0.5)
            results["technique_csv"] = run_technique_evaluation(
                results["shot_csv"], work_dir,
                sync_csv=results["sync_csv"],
                smoothed_csv=results["smoothed_csv"],
            )
        except Exception as e:
            results["errors"]["technique_evaluation"] = str(e)
    else:
        results["errors"]["technique_evaluation"] = "Skipped: no shots were detected."

    # ---------------- FINAL VIDEO OVERLAY ----------------
    if annotated_video and results["shot_csv"]:
        try:
            report(6, 0.0)
            final_path, web_ok, shot_images = run_overlay_final_video(
                annotated_video, results["shot_csv"], work_dir,
                sync_csv=results["sync_csv"],
                technique_csv=results["technique_csv"],
                progress_callback=frame_progress(6),
            )
            results["final_video"] = final_path
            results["video_web_compatible"] = web_ok
            results["shot_images"] = shot_images
            if not web_ok:
                results["errors"]["video_reencode"] = (
                    "Video was generated but could not be re-encoded for inline browser "
                    "playback (ffmpeg step failed or imageio-ffmpeg isn't installed). "
                    "It's still downloadable and will play in VLC/desktop players."
                )
        except Exception as e:
            results["errors"]["final_video"] = f"{e}\n{traceback.format_exc(limit=2)}"
    elif not annotated_video:
        results["errors"]["final_video"] = "Skipped: pose/ball detection pass did not complete."
    else:
        results["errors"]["final_video"] = "Skipped: no shots were detected to annotate."

    # ---------------- SUMMARY ----------------
    if results["technique_csv"]:
        try:
            import pandas as pd
            df = pd.read_csv(results["technique_csv"])
            scores = df["technique_score"].dropna()

            results["summary"] = {
                "total_shots": len(df),
                "average_score": round(float(scores.mean()), 2) if not scores.empty else None,
                "excellent": int((df["overall"] == "EXCELLENT").sum()),
                "good": int((df["overall"] == "GOOD").sum()),
                "average": int((df["overall"] == "AVERAGE").sum()),
                "needs_improvement": int((df["overall"] == "NEEDS IMPROVEMENT").sum()),
            }
        except Exception as e:
            results["errors"]["summary"] = str(e)

    report(7, 1.0)

    return results


def new_work_dir(base_dir):
    """Creates a unique per-run output folder under base_dir."""
    run_id = uuid.uuid4().hex[:10]
    path = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(path, exist_ok=True)
    return path
