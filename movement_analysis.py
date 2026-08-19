import matplotlib
matplotlib.use("Agg")  # CHANGED: non-interactive backend

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CRICKET AI - MOVEMENT ANALYSIS (charts)
# ============================================================
# CHANGED:
#   - matplotlib now uses the "Agg" backend and plt.show() calls
#     were removed. Your original script called plt.show() three
#     times, which opens a blocking GUI window — fine when you
#     run it by hand, but it would hang forever inside app.py or
#     any automated pipeline (nothing is there to close the
#     window). savefig() still produces the same PNG files.
#   - NOT called from pipeline.py / app.py: the Streamlit app
#     builds its own charts directly from the CSVs using
#     st.line_chart, so this script stays available as an
#     optional manual/CLI step rather than adding a redundant
#     chart-generation pass to every analysis run.
# ============================================================

INPUT_FILE = "smoothed_pose_data.csv"


def main():

    print("Loading smoothed pose data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total frames: {len(df)}")

    time = df["time_seconds"]

    # 1. ELBOW ANGLES
    plt.figure(figsize=(12, 6))
    plt.plot(time, df["left_elbow_angle"], label="Left Elbow")
    plt.plot(time, df["right_elbow_angle"], label="Right Elbow")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.title("Elbow Angle During Batting")
    plt.legend()
    plt.grid(True)
    plt.savefig("elbow_movement.png", dpi=150)
    plt.close()

    # 2. KNEE ANGLES
    plt.figure(figsize=(12, 6))
    plt.plot(time, df["left_knee_angle"], label="Left Knee")
    plt.plot(time, df["right_knee_angle"], label="Right Knee")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.title("Knee Angle During Batting")
    plt.legend()
    plt.grid(True)
    plt.savefig("knee_movement.png", dpi=150)
    plt.close()

    # 3. FOOT MOVEMENT
    plt.figure(figsize=(12, 6))
    plt.plot(time, df["foot_distance"], label="Foot Distance")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Distance (pixels)")
    plt.title("Foot Movement During Batting")
    plt.legend()
    plt.grid(True)
    plt.savefig("foot_movement.png", dpi=150)
    plt.close()

    print()
    print("==============================")
    print("MOVEMENT ANALYSIS")
    print("==============================")
    print("\nLeft Elbow:")
    print(f"Minimum: {df['left_elbow_angle'].min():.2f}\u00b0")
    print(f"Maximum: {df['left_elbow_angle'].max():.2f}\u00b0")
    print("\nRight Elbow:")
    print(f"Minimum: {df['right_elbow_angle'].min():.2f}\u00b0")
    print(f"Maximum: {df['right_elbow_angle'].max():.2f}\u00b0")
    print("\nLeft Knee:")
    print(f"Minimum: {df['left_knee_angle'].min():.2f}\u00b0")
    print(f"Maximum: {df['left_knee_angle'].max():.2f}\u00b0")
    print("\nRight Knee:")
    print(f"Minimum: {df['right_knee_angle'].min():.2f}\u00b0")
    print(f"Maximum: {df['right_knee_angle'].max():.2f}\u00b0")
    print()
    print("==============================")
    print("GRAPHS CREATED!")
    print("==============================")
    print("1. elbow_movement.png")
    print("2. knee_movement.png")
    print("3. foot_movement.png")


if __name__ == "__main__":
    main()
