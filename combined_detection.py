import os
import csv
import cv2
import numpy as np

from pose_analysis import calculate_angle, get_point, distance, KEYPOINT_CONFIDENCE

# ============================================================
# CRICKET AI - COMBINED DETECTION PASS
# ============================================================
# NEW FILE — SPEED FIX.
#
# The original pipeline read the video FOUR separate times with
# heavy YOLO models:
#   1. pose_analysis.py     — pose model, whole video
#   2. ball_detection.py    — ball model, whole video
#   3. final_analysis_video — pose model AGAIN, whole video
#   4. final_analysis_video — ball model AGAIN, whole video
#
# This module runs pose model + ball model together in ONE pass
# over the video, producing pose_data.csv, ball_detection.csv, AND
# an annotated video (skeleton + ball drawn) all at once. The
# shot-info text panel (which needs shot_analysis.csv — only
# knowable AFTER this pass, since shot detection looks at the
# whole movement signal) is added afterwards by overlay_final_video.py,
# which does NOT run any model — it just reads frames and draws text,
# so it's fast.
#
# Net effect: 4 full-video model passes -> 1 full-video model pass
# + 1 lightweight text-overlay pass.
#
# Detection thresholds are kept IDENTICAL to your original scripts —
# only where the work happens changed, not what gets detected:
#   - Pose: same PERSON_CONFIDENCE / KEYPOINT_CONFIDENCE as
#     pose_analysis.py, same skeleton-drawing confidence (0.35) as
#     final_analysis_video.py.
#   - Ball: detected at conf=0.15 (same as your original
#     ball_detection.py, so ball_shot_sync still has as much data
#     to match against) but only DRAWN on screen when confidence
#     >= 0.30 (same visual strictness as your original
#     final_analysis_video.py). Nothing about detection sensitivity
#     changed — only where boxes are drawn.
# ============================================================

POSE_CONFIDENCE = 0.45
BALL_DETECT_CONFIDENCE = 0.15   # logged to CSV at this threshold (matches ball_detection.py)
BALL_DRAW_CONFIDENCE = 0.30     # only drawn on screen above this (matches final_analysis_video.py)
BALL_CLASS = 32

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

DRAW_KEYPOINT_CONFIDENCE = 0.35


def run_combined_detection(
    video_path,
    output_dir,
    pose_model,
    ball_model,
    progress_callback=None,
):
    """
    Single pass: pose + ball detection, writes pose_data.csv,
    ball_detection.csv, and an annotated (skeleton+ball) video.

    pose_model / ball_model are already-loaded ultralytics YOLO
    instances (loaded once by pipeline.py and reused, instead of
    each stage reloading its own copy).

    Returns dict: {"pose_csv", "ball_csv", "annotated_video", "frames"}
    """

    os.makedirs(output_dir, exist_ok=True)

    pose_csv_path = os.path.join(output_dir, "pose_data.csv")
    ball_csv_path = os.path.join(output_dir, "ball_detection.csv")
    annotated_video_path = os.path.join(output_dir, "annotated_raw.mp4")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    pose_csv = open(pose_csv_path, mode="w", newline="")
    pose_writer = csv.writer(pose_csv)
    pose_writer.writerow([
        "frame", "time_seconds",
        "left_elbow_angle", "right_elbow_angle",
        "left_knee_angle", "right_knee_angle",
        "left_foot_distance", "right_foot_distance",
        "shoulder_width", "foot_distance",
    ])

    ball_csv = open(ball_csv_path, mode="w", newline="")
    ball_writer = csv.writer(ball_csv)
    ball_writer.writerow([
        "frame", "time_seconds", "ball_detected",
        "ball_x", "ball_y", "confidence",
    ])

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1
        annotated_frame = frame.copy()

        # ---------------- POSE ----------------

        left_elbow_angle = right_elbow_angle = None
        left_knee_angle = right_knee_angle = None
        left_foot_distance = right_foot_distance = None
        shoulder_width = foot_distance = None

        pose_results = pose_model.predict(frame, conf=POSE_CONFIDENCE, verbose=False)
        pose_result = pose_results[0]

        if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:

            keypoints_xy = pose_result.keypoints.xy.cpu().numpy()
            keypoints_conf = (
                pose_result.keypoints.conf.cpu().numpy()
                if pose_result.keypoints.conf is not None else None
            )

            if len(keypoints_xy) > 0:

                person_xy = keypoints_xy[0]
                person_conf = keypoints_conf[0] if keypoints_conf is not None else np.ones(len(person_xy))

                # ---- angle/data extraction (uses KEYPOINT_CONFIDENCE, matches pose_analysis.py) ----
                keypoints_full = np.concatenate(
                    [person_xy, person_conf.reshape(-1, 1)], axis=1
                )

                left_shoulder = get_point(keypoints_full, 5)
                right_shoulder = get_point(keypoints_full, 6)
                left_elbow = get_point(keypoints_full, 7)
                right_elbow = get_point(keypoints_full, 8)
                left_wrist = get_point(keypoints_full, 9)
                right_wrist = get_point(keypoints_full, 10)
                left_hip = get_point(keypoints_full, 11)
                right_hip = get_point(keypoints_full, 12)
                left_knee = get_point(keypoints_full, 13)
                right_knee = get_point(keypoints_full, 14)
                left_ankle = get_point(keypoints_full, 15)
                right_ankle = get_point(keypoints_full, 16)

                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                if left_ankle is not None and right_ankle is not None:
                    foot_distance = distance(left_ankle, right_ankle)
                if left_shoulder is not None and right_shoulder is not None:
                    shoulder_width = distance(left_shoulder, right_shoulder)
                if foot_distance is not None and shoulder_width:
                    left_foot_distance = round(foot_distance / shoulder_width, 2)
                    right_foot_distance = round(foot_distance / shoulder_width, 2)

                # ---- skeleton drawing (uses DRAW_KEYPOINT_CONFIDENCE, matches final_analysis_video.py) ----
                for start, end in SKELETON:
                    if start >= len(person_xy) or end >= len(person_xy):
                        continue
                    if person_conf[start] < DRAW_KEYPOINT_CONFIDENCE or person_conf[end] < DRAW_KEYPOINT_CONFIDENCE:
                        continue
                    x1, y1 = map(int, person_xy[start])
                    x2, y2 = map(int, person_xy[end])
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                for index, point in enumerate(person_xy):
                    if index >= len(person_conf) or person_conf[index] < DRAW_KEYPOINT_CONFIDENCE:
                        continue
                    x, y = map(int, point)
                    cv2.circle(annotated_frame, (x, y), 6, (0, 255, 255), -1)

        pose_writer.writerow([
            frame_number, round(frame_number / fps, 3),
            left_elbow_angle, right_elbow_angle,
            left_knee_angle, right_knee_angle,
            left_foot_distance, right_foot_distance,
            shoulder_width, foot_distance,
        ])

        # ---------------- BALL ----------------

        ball_found = False
        ball_x = ball_y = confidence = None

        ball_results = ball_model(
            frame, classes=[BALL_CLASS], conf=BALL_DETECT_CONFIDENCE, verbose=False,
        )
        ball_result = ball_results[0]

        if ball_result.boxes is not None and len(ball_result.boxes) > 0:

            best_box = max(ball_result.boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            confidence = round(float(best_box.conf[0]), 4)

            ball_x = (x1 + x2) // 2
            ball_y = (y1 + y2) // 2
            ball_found = True

            if confidence >= BALL_DRAW_CONFIDENCE:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.circle(annotated_frame, (ball_x, ball_y), 8, (0, 0, 255), -1)
                cv2.putText(
                    annotated_frame, f"BALL {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

        ball_writer.writerow([
            frame_number, round(frame_number / fps, 3),
            ball_found, ball_x, ball_y, confidence,
        ])

        out.write(annotated_frame)

        if progress_callback and frame_number % 10 == 0:
            progress_callback(frame_number, total_frames)

    cap.release()
    out.release()
    pose_csv.close()
    ball_csv.close()

    return {
        "pose_csv": pose_csv_path,
        "ball_csv": ball_csv_path,
        "annotated_video": annotated_video_path,
        "frames": frame_number,
    }
