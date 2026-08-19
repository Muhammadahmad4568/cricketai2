import os
import subprocess

# ============================================================
# CRICKET AI - VIDEO UTILITIES
# ============================================================
# NEW FILE.
#
# WHY THIS EXISTS:
# cv2.VideoWriter with the "mp4v" fourcc (what every script in this
# project used) produces a valid .mp4 file, but it's encoded as
# MPEG-4 Part 2 — most browsers (including the one Streamlit's
# st.video() uses) only reliably play H.264 ("avc1") inside an mp4
# container. That's why the final video showed up fine in the
# runs/ folder (any desktop video player handles mp4v) but wouldn't
# play inline in Streamlit.
#
# FIX: after writing a video with cv2 as before, re-encode it to
# H.264 using a bundled ffmpeg binary (via the imageio-ffmpeg
# package — no need to separately install ffmpeg on Windows).
# If that's not available for any reason, we fail soft and return
# the original file so the pipeline doesn't crash — the video will
# still be downloadable and playable in VLC/desktop players even if
# it won't preview inline.
# ============================================================


def reencode_for_browser(input_path, output_path=None):
    """
    Re-encodes input_path to H.264/yuv420p so it plays inline in
    Streamlit / any browser. Returns the path to the playable file.

    On any failure (ffmpeg missing, conversion error), logs a warning
    and returns the original input_path unchanged instead of raising —
    analysis should not fail just because browser playback re-encoding
    didn't work.
    """

    if output_path is None:
        base, _ext = os.path.splitext(input_path)
        output_path = f"{base}_web.mp4"

    try:
        import imageio_ffmpeg
    except ImportError:
        print(
            "[video_utils] imageio-ffmpeg not installed — skipping browser "
            "re-encode. Run: pip install imageio-ffmpeg"
        )
        return input_path, False

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe, "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0 or not os.path.exists(output_path):
            print("[video_utils] ffmpeg re-encode failed:\n" + result.stderr[-1000:])
            return input_path, False

        return output_path, True

    except Exception as e:
        print(f"[video_utils] Browser re-encode skipped due to error: {e}")
        return input_path, False
