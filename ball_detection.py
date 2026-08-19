import os
import csv
from ultralytics import YOLO
import cv2

# ============================================================
# CRICKET AI - BALL DETECTION
# ============================================================
# CHANGED FOR PIPELINE INTEGRATION:
#   - INTEGRATION FIX: your original script only drew boxes onto
#     an output video and threw the coordinates away afterward.
#     ball_shot_sync needs per-frame ball positions/timestamps to
#     match against shot moments, so this version ALSO writes
#     ball_detection.csv (frame, time_seconds, ball_detected,
#     ball_x, ball_y, confidence) using the exact same detection
#     call you already had (same model file, same class ID 32,
#     same 0.15 confidence threshold). The detector itself is
#     untouched — only the missing "save what we found" step
#     was added.
#   - Wrapped in run_ball_detection(video_path, output_dir, ...)
#     instead of hardcoded VIDEO_FILE / OUTPUT_FILE constants.
#   - save_video defaults to False in the pipeline: the final
#     combined video (final_analysis_video.py) already re-detects
#     and draws the ball, so writing a second annotated video here
#     is redundant work. It's still available via save_video=True
#     for standalone/CLI use.
# ============================================================

MODEL_FILE_DEFAULT = "yolov8n.pt"

# COCO dataset "sports ball" class ID
SPORTS_BALL_CLASS = 32
BALL_CONFIDENCE = 0.15


def run_ball_detection(
    video_path,
    output_dir,
    model_path=MODEL_FILE_DEFAULT,
    save_video=False,
    progress_callback=None,
):
    """
    Runs the sports-ball detector over video_path and writes
    ball_detection.csv (one row per frame) into output_dir.
    Optionally also writes an annotated ball_detection.mp4.

    Returns dict: {"csv_path": ..., "video_path": ... or None,
                    "frames": int, "detections": int}
    """

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "ball_detection.csv")
    output_video = os.path.join(output_dir, "ball_detection.mp4")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    csv_file = open(output_csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "time_seconds", "ball_detected",
        "ball_x", "ball_y", "confidence",
    ])

    frame_number = 0
    detections = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model(
            frame,
            classes=[SPORTS_BALL_CLASS],
            conf=BALL_CONFIDENCE,
            verbose=False,
        )

        result = results[0]

        ball_found = False
        ball_x = ball_y = confidence = None

        if result.boxes is not None and len(result.boxes) > 0:

            # Keep the highest-confidence detection in this frame
            # (your original script drew every box; for the sync CSV
            # we need exactly one ball position per frame).
            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))

            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            confidence = round(float(best_box.conf[0]), 4)

            ball_x = (x1 + x2) // 2
            ball_y = (y1 + y2) // 2
            ball_found = True
            detections += 1

            if save_video:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (ball_x, ball_y), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"BALL {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

        csv_writer.writerow([
            frame_number,
            round(frame_number / fps, 3),
            ball_found,
            ball_x, ball_y, confidence,
        ])

        if save_video and out is not None:
            out.write(frame)

        if progress_callback and frame_number % 10 == 0:
            progress_callback(frame_number, total_frames)

    cap.release()

    if out is not None:
        out.release()

    csv_file.close()

    return {
        "csv_path": output_csv,
        "video_path": output_video if save_video else None,
        "frames": frame_number,
        "detections": detections,
    }


# ============================================================
# CLI MODE (unchanged behavior: run against batting_video.mp4)
# ============================================================

if __name__ == "__main__":

    print("Loading YOLO model...")

    result = run_ball_detection(
        video_path="batting_video.mp4",
        output_dir=".",
        save_video=True,
    )

    print()
    print("==============================")
    print("BALL DETECTION COMPLETE!")
    print("==============================")
    print(f"Output video: {result['video_path']}")
    print(f"Output CSV: {result['csv_path']}")
    print(f"Frames processed: {result['frames']}")
    print(f"Ball detections: {result['detections']}")
