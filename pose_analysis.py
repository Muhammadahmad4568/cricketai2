import cv2
import csv
import math
import os
from ultralytics import YOLO


# ============================================================
# CRICKET AI - POSE ANALYSIS
# ============================================================
# CHANGED FOR PIPELINE INTEGRATION:
#   - Wrapped in run_pose_analysis(video_path, output_dir, ...)
#     instead of hardcoded INPUT_VIDEO / OUTPUT_* constants, so
#     it works on any uploaded video instead of only
#     "batting_video.mp4".
#   - save_video is optional (default True) so pipeline.py can
#     skip writing the intermediate annotated pose video when
#     it isn't needed (the final combined video already draws
#     the skeleton), saving one full video-encode pass.
#   - Everything else (model, thresholds, angle/keypoint logic)
#     is unchanged from your original script.
# ============================================================

MODEL_PATH_DEFAULT = "yolov8n-pose.pt"

PERSON_CONFIDENCE = 0.5
KEYPOINT_CONFIDENCE = 0.30


def calculate_angle(a, b, c):
    """Calculates angle ABC using three points."""
    if None in (a, b, c):
        return None

    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]

    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = max(-1, min(1, cosine_angle))

    angle = math.degrees(math.acos(cosine_angle))

    return round(angle, 2)


def get_point(keypoints, index):
    """Returns a keypoint only if its confidence is good enough."""
    x, y = keypoints[index][:2]
    confidence = keypoints[index][2]

    if confidence < KEYPOINT_CONFIDENCE:
        return None

    return (float(x), float(y))


def distance(p1, p2):
    if p1 is None or p2 is None:
        return None

    return round(
        math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2),
        2
    )


def run_pose_analysis(
    video_path,
    output_dir,
    model_path=MODEL_PATH_DEFAULT,
    save_video=True,
    progress_callback=None,
):
    """
    Runs YOLO pose detection over video_path and writes pose_data.csv
    (and optionally an annotated pose_analysis.mp4) into output_dir.

    Returns dict: {"csv_path": ..., "video_path": ... or None, "frames": int}
    """

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "pose_data.csv")
    output_video = os.path.join(output_dir, "pose_analysis.mp4")

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
        "frame", "time_seconds",
        "left_elbow_angle", "right_elbow_angle",
        "left_knee_angle", "right_knee_angle",
        "left_foot_distance", "right_foot_distance",
        "shoulder_width", "foot_distance",
    ])

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.predict(frame, conf=PERSON_CONFIDENCE, verbose=False)
        result = results[0]

        left_elbow_angle = right_elbow_angle = None
        left_knee_angle = right_knee_angle = None
        left_foot_distance = right_foot_distance = None
        shoulder_width = foot_distance = None

        annotated_frame = frame
        has_detection = False

        if result.keypoints is not None and len(result.keypoints) > 0:

            has_detection = True
            person_index = 0
            keypoints = result.keypoints.data[person_index].cpu().numpy()

            left_shoulder = get_point(keypoints, 5)
            right_shoulder = get_point(keypoints, 6)
            left_elbow = get_point(keypoints, 7)
            right_elbow = get_point(keypoints, 8)
            left_wrist = get_point(keypoints, 9)
            right_wrist = get_point(keypoints, 10)
            left_hip = get_point(keypoints, 11)
            right_hip = get_point(keypoints, 12)
            left_knee = get_point(keypoints, 13)
            right_knee = get_point(keypoints, 14)
            left_ankle = get_point(keypoints, 15)
            right_ankle = get_point(keypoints, 16)

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

            if save_video:
                annotated_frame = result.plot()

        if save_video and out is not None:
            out.write(annotated_frame)

        time_seconds = round(frame_number / fps, 3)

        csv_writer.writerow([
            frame_number, time_seconds,
            left_elbow_angle, right_elbow_angle,
            left_knee_angle, right_knee_angle,
            left_foot_distance, right_foot_distance,
            shoulder_width, foot_distance,
        ])

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
    }


# ============================================================
# CLI MODE (unchanged behavior: run against batting_video.mp4
# in the current directory, matching your original script)
# ============================================================

if __name__ == "__main__":

    print("Loading YOLO Pose model...")

    result = run_pose_analysis(
        video_path="batting_video.mp4",
        output_dir=".",
        save_video=True,
    )

    print()
    print("================================")
    print("POSE ANALYSIS COMPLETE!")
    print("================================")
    print(f"Video: {result['video_path']}")
    print(f"Data:  {result['csv_path']}")
