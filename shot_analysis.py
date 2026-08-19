import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# ============================================================
# CRICKET AI - SHOT ANALYSIS
# ============================================================
# CHANGED FOR PIPELINE INTEGRATION:
#   - Wrapped in run_shot_analysis(input_csv, output_dir) instead
#     of hardcoded INPUT_FILE / OUTPUT_FILE constants.
#   - Column-detection, movement-signal, and peak-finding logic
#     are UNCHANGED from your original script.
# ============================================================


def find_column(df, possible_names):

    for name in possible_names:
        if name in df.columns:
            return name

    for col in df.columns:
        col_lower = col.lower()
        for name in possible_names:
            if name.lower() in col_lower:
                return col

    return None


def run_shot_analysis(input_csv, output_dir):
    """
    Detects shot moments from smoothed_pose_data.csv using movement-peak
    detection and writes shot_analysis.csv into output_dir.

    Returns the output CSV path.
    """

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "shot_analysis.csv")

    df = pd.read_csv(input_csv)

    # --------------------------------------------------------
    # TIME COLUMN
    # --------------------------------------------------------

    time_col = None

    for col in ["time", "Time", "timestamp", "Timestamp", "time_seconds"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        df["time_seconds"] = np.arange(len(df)) / 30.0
        time_col = "time_seconds"

    # --------------------------------------------------------
    # RELEVANT COLUMNS
    # --------------------------------------------------------

    left_elbow = find_column(df, ["left_elbow_angle", "left_elbow"])
    right_elbow = find_column(df, ["right_elbow_angle", "right_elbow"])
    left_knee = find_column(df, ["left_knee_angle", "left_knee"])
    right_knee = find_column(df, ["right_knee_angle", "right_knee"])
    left_ankle_x = find_column(df, ["left_ankle_x"])
    right_ankle_x = find_column(df, ["right_ankle_x"])
    left_ankle_y = find_column(df, ["left_ankle_y"])
    right_ankle_y = find_column(df, ["right_ankle_y"])

    # --------------------------------------------------------
    # MOVEMENT SIGNAL
    # --------------------------------------------------------

    df["movement_score"] = 0.0

    signals = []

    for col in [left_elbow, right_elbow, left_knee, right_knee,
                left_ankle_x, right_ankle_x, left_ankle_y, right_ankle_y]:

        if col is not None:
            values = pd.to_numeric(df[col], errors="coerce").interpolate().bfill().ffill()
            signals.append(values.diff().abs())

    if signals:
        movement_matrix = pd.concat(signals, axis=1)
        df["movement_score"] = movement_matrix.mean(axis=1)

    df["movement_smooth"] = (
        df["movement_score"]
        .rolling(window=7, center=True)
        .mean()
        .bfill()
        .ffill()
    )

    # --------------------------------------------------------
    # PEAK DETECTION
    # --------------------------------------------------------

    signal = df["movement_smooth"].values

    mean_movement = np.mean(signal)
    std_movement = np.std(signal)

    threshold = mean_movement + (1.2 * std_movement)
    min_distance = 25

    peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)

    # --------------------------------------------------------
    # SHOT TABLE
    # --------------------------------------------------------

    shots = []

    for shot_number, frame_index in enumerate(peaks, start=1):

        row = df.iloc[frame_index]

        # BUG FIX: frame_index here is find_peaks' POSITIONAL array index
        # (0-based), not the actual video frame number. It was being stored
        # directly as "frame", which is off by however many rows were
        # dropped/reindexed before this point (in practice ~1, since
        # pose_data.csv's own "frame" column starts at 1). The ±12-frame
        # tolerance used everywhere this gets matched against (final video
        # panel, overlay image capture) absorbed that small drift silently,
        # but it matters once we need to grab an EXACT frame for the shot
        # highlight image below — so we now store the real frame number
        # from the row's own "frame" column when it exists.
        actual_frame = int(row["frame"]) if "frame" in df.columns else int(frame_index)

        shot_time = float(row[time_col])
        movement_value = float(row["movement_smooth"])

        elbow_values = []
        for col in [left_elbow, right_elbow]:
            if col is not None:
                value = pd.to_numeric(row[col], errors="coerce")
                if not pd.isna(value):
                    elbow_values.append(float(value))

        knee_values = []
        for col in [left_knee, right_knee]:
            if col is not None:
                value = pd.to_numeric(row[col], errors="coerce")
                if not pd.isna(value):
                    knee_values.append(float(value))

        average_elbow = np.mean(elbow_values) if elbow_values else np.nan
        average_knee = np.mean(knee_values) if knee_values else np.nan

        shots.append({
            "shot_number": shot_number,
            "frame": actual_frame,
            "time_seconds": round(shot_time, 3),
            "movement_score": round(movement_value, 3),
            "average_elbow_angle": round(average_elbow, 2) if not np.isnan(average_elbow) else None,
            "average_knee_angle": round(average_knee, 2) if not np.isnan(average_knee) else None,
        })

    shots_df = pd.DataFrame(shots, columns=[
        "shot_number", "frame", "time_seconds", "movement_score",
        "average_elbow_angle", "average_knee_angle",
    ])

    shots_df.to_csv(output_csv, index=False)

    return output_csv


# ============================================================
# CLI MODE (unchanged behavior)
# ============================================================

if __name__ == "__main__":

    print("Loading pose data...")

    path = run_shot_analysis("smoothed_pose_data.csv", ".")

    result_df = pd.read_csv(path)

    print()
    print("========================================")
    print("       SHOT ANALYSIS COMPLETE")
    print("========================================")
    print(f"Output file: {path}")
    print(f"Shots detected: {len(result_df)}")
