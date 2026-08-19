import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# ============================================================
# CRICKET AI - FINAL ANALYSIS VIDEO
# ============================================================
# CHANGED FOR PIPELINE INTEGRATION:
#   - Wrapped in run_final_video(...) instead of hardcoded
#     VIDEO_FILE / SHOT_FILE / SYNC_FILE / OUTPUT_FILE constants,
#     so it works on any uploaded video + freshly generated CSVs.
#   - sync_csv is now OPTIONAL: your original script hard-required
#     ball_shot_sync.csv and would crash (raise SystemExit) if it
#     was missing. Since that file didn't exist anywhere in your
#     upload, this version degrades gracefully — it still draws
#     pose + ball + shot info, just without the ball-sync overlay,
#     and shows "N/A" instead of crashing.
#   - Optional technique_csv adds the technique score/rating to
#     the on-screen shot panel when available.
#   - Detection logic (pose skeleton drawing, ball box drawing,
#     panel layout) is otherwise UNCHANGED from your original.
# ============================================================

POSE_MODEL_DEFAULT = "yolov8n-pose.pt"
BALL_MODEL_DEFAULT = "yolov8n.pt"

POSE_CONFIDENCE = 0.45
BALL_CONFIDENCE = 0.30

BALL_CLASS = 32

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def run_final_video(
    video_path,
    shot_csv,
    output_dir,
    sync_csv=None,
    technique_csv=None,
    pose_model_path=POSE_MODEL_DEFAULT,
    ball_model_path=BALL_MODEL_DEFAULT,
    progress_callback=None,
):
    """
    Produces the annotated final analysis video: pose skeleton,
    ball detection, and a shot-info panel overlaid on the original
    video. Returns the output video path.
    """

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_cricket_analysis.mp4")

    shot_df = pd.read_csv(shot_csv) if os.path.exists(shot_csv) else pd.DataFrame()

    sync_df = pd.DataFrame()
    if sync_csv and os.path.exists(sync_csv):
        sync_df = pd.read_csv(sync_csv)

    technique_df = pd.DataFrame()
    if technique_csv and os.path.exists(technique_csv):
        technique_df = pd.read_csv(technique_csv)

    pose_model = YOLO(pose_model_path)
    ball_model = YOLO(ball_model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

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
        if match.empty:
            return None
        return match.iloc[0]

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ---------------- POSE ----------------

        pose_results = pose_model.predict(frame, conf=POSE_CONFIDENCE, verbose=False)
        annotated_frame = frame.copy()

        for result in pose_results:

            if result.keypoints is None:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None else None
            )

            if len(keypoints) == 0:
                continue

            person = keypoints[0]
            person_conf = confidences[0] if confidences is not None else np.ones(len(person))

            for start, end in SKELETON:
                if start >= len(person) or end >= len(person):
                    continue
                if person_conf[start] < 0.35 or person_conf[end] < 0.35:
                    continue
                x1, y1 = map(int, person[start])
                x2, y2 = map(int, person[end])
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            for index, point in enumerate(person):
                if index >= len(person_conf) or person_conf[index] < 0.35:
                    continue
                x, y = map(int, point)
                cv2.circle(annotated_frame, (x, y), 6, (0, 255, 255), -1)

        # ---------------- BALL ----------------

        ball_results = ball_model.predict(frame, classes=[BALL_CLASS], conf=BALL_CONFIDENCE, verbose=False)

        ball_found = False
        ball_conf = None

        for result in ball_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = map(int, xyxy)
                ball_x = int((x1 + x2) / 2)
                ball_y = int((y1 + y2) / 2)
                ball_conf = confidence
                ball_found = True

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.circle(annotated_frame, (ball_x, ball_y), 8, (0, 0, 255), -1)
                cv2.putText(
                    annotated_frame, f"BALL {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
                break
            if ball_found:
                break

        # ---------------- SHOT INFO ----------------

        shot = get_nearest_shot(frame_number)

        # ---------------- TOP PANEL ----------------

        cv2.rectangle(annotated_frame, (0, 0), (width, 90), (30, 30, 30), -1)
        cv2.putText(
            annotated_frame, "CRICKET AI - BATTING ANALYSIS", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )

        time_seconds = frame_number / fps
        cv2.putText(
            annotated_frame, f"Time: {time_seconds:.2f}s", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )

        # ---------------- SHOT PANEL ----------------

        if shot is not None:

            shot_number = int(shot["shot_number"])
            elbow = shot.get("average_elbow_angle")
            knee = shot.get("average_knee_angle")
            movement = shot.get("movement_score")

            technique = get_technique_for_shot(shot_number)

            panel_y = height - 165 if technique is not None else height - 145

            cv2.rectangle(annotated_frame, (10, panel_y), (430, height - 10), (30, 30, 30), -1)
            cv2.putText(
                annotated_frame, f"SHOT #{shot_number}", (25, panel_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

            elbow_text = f"Elbow: {elbow:.1f} deg" if pd.notna(elbow) else "Elbow: N/A"
            knee_text = f"Knee: {knee:.1f} deg" if pd.notna(knee) else "Knee: N/A"

            cv2.putText(annotated_frame, elbow_text, (25, panel_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_frame, knee_text, (25, panel_y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if pd.notna(movement):
                cv2.putText(annotated_frame, f"Movement: {movement:.2f}", (25, panel_y + 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if technique is not None:
                score = technique.get("technique_score")
                overall = technique.get("overall")
                score_text = (
                    f"Score: {score:.1f} ({overall})"
                    if pd.notna(score) else f"Score: N/A ({overall})"
                )
                cv2.putText(annotated_frame, score_text, (25, panel_y + 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2)

        # ---------------- BALL INFO ----------------

        if ball_found:
            cv2.putText(
                annotated_frame, f"Ball detected: {ball_conf:.2f}", (width - 300, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        out.write(annotated_frame)
        frame_number += 1

        if progress_callback and frame_number % 10 == 0:
            progress_callback(frame_number, total_frames)

    cap.release()
    out.release()

    return output_file


# ============================================================
# CLI MODE
# ============================================================

if __name__ == "__main__":

    print("Generating final analysis video...")

    result = run_final_video(
        video_path="batting_video.mp4",
        shot_csv="shot_analysis.csv",
        output_dir=".",
        sync_csv="ball_shot_sync.csv" if os.path.exists("ball_shot_sync.csv") else None,
        technique_csv="technique_evaluation.csv" if os.path.exists("technique_evaluation.csv") else None,
    )

    print()
    print("========================================")
    print("FINAL VIDEO COMPLETE")
    print("========================================")
    print(f"Output saved as: {result}")
