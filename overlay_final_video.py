import os
import cv2
import pandas as pd

from video_utils import reencode_for_browser

# ============================================================
# CRICKET AI - FINAL OVERLAY PASS
# ============================================================
# NEW FILE — pairs with combined_detection.py for the speed fix.
#
# This reads the already-annotated video (skeleton + ball, produced
# by combined_detection.py) and adds the top info bar + per-shot
# panel — the same overlay your original final_analysis_video.py
# drew, just without re-running any YOLO model, since the skeleton
# and ball boxes are already baked into the frames. This pass is
# just image read/write + text drawing, so it's fast even on CPU.
#
# At the end it re-encodes to H.264 via video_utils.reencode_for_browser
# so the result actually plays inline in Streamlit (mp4v does not,
# reliably, in most browsers).
# ============================================================


def run_overlay_final_video(
    annotated_video_path,
    shot_csv,
    output_dir,
    sync_csv=None,
    technique_csv=None,
    progress_callback=None,
):
    """
    Adds the shot-info overlay panel to an already pose/ball-annotated
    video, saves one highlight image per shot (the annotated frame at
    that shot's exact moment), and returns:
      (final_video_path, web_compatible_bool, shot_images_dict)
    where shot_images_dict maps shot_number -> saved image path.
    """

    os.makedirs(output_dir, exist_ok=True)
    raw_output = os.path.join(output_dir, "final_cricket_analysis_raw.mp4")

    shots_dir = os.path.join(output_dir, "shot_frames")
    os.makedirs(shots_dir, exist_ok=True)

    shot_df = pd.read_csv(shot_csv) if os.path.exists(shot_csv) else pd.DataFrame()

    technique_df = pd.DataFrame()
    if technique_csv and os.path.exists(technique_csv):
        technique_df = pd.read_csv(technique_csv)

    # Exact frame -> shot_number, for capturing one highlight image per shot.
    # (Now reliable since shot_analysis.py stores the real video frame
    # number, not a positional array index — see the fix there.)
    capture_targets = {}
    if not shot_df.empty:
        for _, row in shot_df.iterrows():
            capture_targets[int(row["frame"])] = int(row["shot_number"])

    shot_images = {}

    cap = cv2.VideoCapture(annotated_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open annotated video: {annotated_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_output, fourcc, fps, (width, height))

    def get_nearest_shot(frame_number):
        if shot_df.empty:
            return None
        differences = (shot_df["frame"] - frame_number).abs()
        nearest_index = differences.idxmin()
        if differences.loc[nearest_index] <= 12:
            return shot_df.loc[nearest_index]
        return None

    def get_technique_for_shot(shot_number):
        if technique_df.empty:
            return None
        match = technique_df[technique_df["shot_number"] == shot_number]
        return match.iloc[0] if not match.empty else None

    # CHANGED: frame_number now increments right after read (1-based),
    # matching combined_detection.py's numbering — previously this loop
    # counted 0-based while shot_analysis.csv used 1-based frame numbers,
    # a mismatch that ±12-frame tolerance hid for the text panel but
    # would have made exact-frame image capture miss by one.
    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ---------------- TOP PANEL ----------------

        cv2.rectangle(frame, (0, 0), (width, 90), (30, 30, 30), -1)
        cv2.putText(
            frame, "CRICKET AI - BATTING ANALYSIS", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Time: {frame_number / fps:.2f}s", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )

        # ---------------- SHOT PANEL ----------------

        shot = get_nearest_shot(frame_number)

        if shot is not None:

            shot_number = int(shot["shot_number"])
            elbow = shot.get("average_elbow_angle")
            knee = shot.get("average_knee_angle")
            movement = shot.get("movement_score")

            technique = get_technique_for_shot(shot_number)
            panel_y = height - 165 if technique is not None else height - 145

            cv2.rectangle(frame, (10, panel_y), (430, height - 10), (30, 30, 30), -1)
            cv2.putText(frame, f"SHOT #{shot_number}", (25, panel_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elbow_text = f"Elbow: {elbow:.1f} deg" if pd.notna(elbow) else "Elbow: N/A"
            knee_text = f"Knee: {knee:.1f} deg" if pd.notna(knee) else "Knee: N/A"

            cv2.putText(frame, elbow_text, (25, panel_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, knee_text, (25, panel_y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if pd.notna(movement):
                cv2.putText(frame, f"Movement: {movement:.2f}", (25, panel_y + 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if technique is not None:
                score = technique.get("technique_score")
                overall = technique.get("overall")
                score_text = (
                    f"Score: {score:.1f} ({overall})" if pd.notna(score) else f"Score: N/A ({overall})"
                )
                cv2.putText(frame, score_text, (25, panel_y + 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2)

        # ---------------- HIGHLIGHT IMAGE CAPTURE ----------------
        # Save the fully-annotated frame (skeleton + ball + panel already
        # drawn) at this shot's exact moment — this is what shows up as
        # the "best shot" highlight image in the report.

        if frame_number in capture_targets:
            shot_num = capture_targets[frame_number]
            image_path = os.path.join(shots_dir, f"shot_{shot_num}.jpg")
            cv2.imwrite(image_path, frame)
            shot_images[shot_num] = image_path

        out.write(frame)

        if progress_callback and frame_number % 20 == 0:
            progress_callback(frame_number, total_frames)

    cap.release()
    out.release()

    final_path, reencoded = reencode_for_browser(
        raw_output, os.path.join(output_dir, "final_cricket_analysis.mp4")
    )

    if reencoded and final_path != raw_output and os.path.exists(raw_output):
        try:
            os.remove(raw_output)
        except OSError:
            pass

    return final_path, reencoded, shot_images
