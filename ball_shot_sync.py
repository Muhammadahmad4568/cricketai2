import os
import pandas as pd

# ============================================================
# CRICKET AI - BALL / SHOT SYNCHRONIZATION
# ============================================================
# NEW FILE — this did not exist in your uploaded project.
# Your brief and app.py both reference ball_shot_sync.csv, but
# no script produced it. This is a straightforward first version:
#
# For each detected shot (from shot_analysis.csv), look at ball
# detections (from ball_detection.csv, now produced by the
# updated ball_detection.py) within a small window of frames
# around the shot moment, and pick the closest one.
#
# SYNC_WINDOW_FRAMES is a tunable constant, not a measured value —
# it approximates "ball should be near the bat within about half
# a second of the swing peak" at a typical 30fps clip. Adjust it
# once you have real footage to check against.
# ============================================================

SYNC_WINDOW_FRAMES = 15


def run_ball_shot_sync(shot_csv, ball_csv, output_dir, window_frames=SYNC_WINDOW_FRAMES):
    """
    Matches each shot to the nearest ball detection within
    window_frames and writes ball_shot_sync.csv into output_dir.

    Returns the output CSV path, or None if inputs are missing/empty.
    """

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "ball_shot_sync.csv")

    if not os.path.exists(shot_csv) or not os.path.exists(ball_csv):
        return None

    shots_df = pd.read_csv(shot_csv)
    ball_df = pd.read_csv(ball_csv)

    if shots_df.empty:
        pd.DataFrame(columns=[
            "shot_number", "shot_frame", "ball_frame", "ball_detected",
            "ball_confidence", "frame_difference", "ball_time_seconds",
        ]).to_csv(output_csv, index=False)
        return output_csv

    detected_balls = ball_df[ball_df["ball_detected"] == True].copy()  # noqa: E712

    rows = []

    for _, shot in shots_df.iterrows():

        shot_number = shot["shot_number"]
        shot_frame = shot["frame"]

        row = {
            "shot_number": shot_number,
            "shot_frame": shot_frame,
            "ball_frame": None,
            "ball_detected": False,
            "ball_confidence": None,
            "frame_difference": None,
            "ball_time_seconds": None,
        }

        if not detected_balls.empty:

            diffs = (detected_balls["frame"] - shot_frame).abs()
            nearest_idx = diffs.idxmin()
            nearest_diff = diffs.loc[nearest_idx]

            if nearest_diff <= window_frames:
                nearest = detected_balls.loc[nearest_idx]
                row["ball_frame"] = int(nearest["frame"])
                row["ball_detected"] = True
                row["ball_confidence"] = nearest["confidence"]
                row["frame_difference"] = int(nearest_diff)
                row["ball_time_seconds"] = nearest["time_seconds"]

        rows.append(row)

    sync_df = pd.DataFrame(rows)
    sync_df.to_csv(output_csv, index=False)

    return output_csv


# ============================================================
# CLI MODE
# ============================================================

if __name__ == "__main__":

    print("Syncing shots with ball detections...")

    path = run_ball_shot_sync("shot_analysis.csv", "ball_detection.csv", ".")

    if path:
        result_df = pd.read_csv(path)
        matched = int(result_df["ball_detected"].sum())
        print()
        print("========================================")
        print("   BALL-SHOT SYNC COMPLETE")
        print("========================================")
        print(f"Output file: {path}")
        print(f"Shots matched to a ball detection: {matched}/{len(result_df)}")
    else:
        print("Missing shot_analysis.csv or ball_detection.csv — nothing to sync.")
