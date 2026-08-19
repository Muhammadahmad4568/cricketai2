import os
import pandas as pd

# ============================================================
# CRICKET AI - POSE SMOOTHING
# ============================================================
# CHANGED FOR PIPELINE INTEGRATION:
#   - Wrapped in run_pose_smoothing(input_csv, output_dir) instead
#     of hardcoded INPUT_FILE / OUTPUT_FILE constants.
#   - Smoothing logic (7-frame centered rolling mean) is unchanged.
# ============================================================

WINDOW_SIZE = 7

COLUMNS_TO_SMOOTH = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "foot_distance",
    "shoulder_width",
]


def run_pose_smoothing(input_csv, output_dir, window_size=WINDOW_SIZE):
    """
    Applies a centered rolling-average smoothing to pose_data.csv
    and writes smoothed_pose_data.csv into output_dir.

    Returns the output CSV path.
    """

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "smoothed_pose_data.csv")

    df = pd.read_csv(input_csv)

    for column in COLUMNS_TO_SMOOTH:
        if column in df.columns:
            df[column] = (
                df[column]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )

    for column in COLUMNS_TO_SMOOTH:
        if column in df.columns:
            df[column] = df[column].round(2)

    df.to_csv(output_csv, index=False)

    return output_csv


# ============================================================
# CLI MODE (unchanged behavior)
# ============================================================

if __name__ == "__main__":

    print("Loading pose data...")

    path = run_pose_smoothing("pose_data.csv", ".")

    print()
    print("==============================")
    print("POSE SMOOTHING COMPLETE!")
    print("==============================")
    print(f"Output: {path}")
