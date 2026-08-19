import os
import uuid
import traceback

import streamlit as st
import pandas as pd

from pipeline import run_analysis, new_work_dir, STAGE_LABELS

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CricketAI | Batting Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_videos")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

POSE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-pose.pt")
BALL_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


# ============================================================
# PREMIUM UI (dark futuristic sports-tech theme)
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(0,255,170,.09), transparent 25%),
        radial-gradient(circle at 100% 10%, rgba(0,130,255,.08), transparent 25%),
        #06101a;
    color: #ffffff;
}

section[data-testid="stSidebar"] {
    background: #07131f;
    border-right: 1px solid rgba(255,255,255,.06);
}

.hero {
    padding: 42px;
    border-radius: 26px;
    background:
        linear-gradient(135deg, rgba(0,255,170,.12), rgba(0,120,255,.08)),
        rgba(10,24,38,.92);
    border: 1px solid rgba(0,255,170,.15);
    margin-bottom: 25px;
}

.hero h1 { font-size: 46px; font-weight: 800; margin: 5px 0; }
.hero p { color: #8da3b4; font-size: 16px; }

.eyebrow {
    color: #00ffaa;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.card {
    background: rgba(10,25,39,.88);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 15px;
}

.metric {
    background: linear-gradient(145deg,#102234,#091622);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 18px;
    padding: 22px;
    min-height: 120px;
}

.metric-title {
    color: #8096a7;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.metric-value { color: white; font-size: 32px; font-weight: 800; margin-top: 8px; }

.section { font-size: 25px; font-weight: 800; margin-top: 35px; margin-bottom: 16px; }

.upload-box {
    padding: 15px;
    border-radius: 20px;
    border: 1px dashed rgba(0,255,170,.25);
    background: rgba(0,255,170,.025);
}

.stButton > button { border-radius: 12px; font-weight: 700; }

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}

.badge-excellent { background: rgba(0,255,170,.15); color: #00ffaa; }
.badge-good { background: rgba(0,180,255,.15); color: #4db8ff; }
.badge-average { background: rgba(255,190,0,.15); color: #ffbe00; }
.badge-needs { background: rgba(255,80,80,.15); color: #ff5050; }
.badge-na { background: rgba(255,255,255,.08); color: #8096a7; }

.warn-box {
    border: 1px solid rgba(255,190,0,.3);
    background: rgba(255,190,0,.06);
    border-radius: 14px;
    padding: 14px 18px;
    color: #ffdf99;
    font-size: 13px;
    margin-bottom: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def badge(label):
    css_map = {
        "EXCELLENT": "badge-excellent",
        "GOOD": "badge-good",
        "AVERAGE": "badge-average",
        "NEEDS IMPROVEMENT": "badge-needs",
    }
    css_class = css_map.get(str(label).upper(), "badge-na")
    return f'<span class="badge {css_class}">{label}</span>'


def load_csv_safe(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_text_report(results):
    """Plain-text summary report built from the real run results, for download."""

    summary = results.get("summary")
    technique_df = load_csv_safe(results.get("technique_csv"))

    lines = ["CRICKETAI - BATTING TECHNIQUE REPORT", "=" * 40, ""]

    if summary:
        score_display = summary["average_score"] if summary["average_score"] is not None else "N/A"
        lines += [
            f"Average Technique Score: {score_display}/100",
            f"Total Shots Analyzed: {summary['total_shots']}",
            f"Excellent: {summary['excellent']}",
            f"Good: {summary['good']}",
            f"Average: {summary['average']}",
            f"Needs Improvement: {summary['needs_improvement']}",
            "",
        ]

    if not technique_df.empty:
        lines.append("SHOT-BY-SHOT BREAKDOWN")
        lines.append("-" * 40)
        for _, row in technique_df.iterrows():
            score = f"{row['technique_score']:.2f}" if pd.notna(row["technique_score"]) else "N/A"
            lines.append(
                f"Shot #{int(row['shot_number'])} | Score: {score}/100 ({row['overall']}) | "
                f"Elbow: {row['elbow']} | Knee: {row['knee']} | Timing: {row['timing']}"
            )
            if pd.notna(row.get("advice")):
                lines.append(f"   Advice: {row['advice']}")
        lines.append("")

    errors = results.get("errors") or {}
    if errors:
        lines.append("NOTES / STAGES THAT DID NOT COMPLETE")
        lines.append("-" * 40)
        for stage, err in errors.items():
            lines.append(f"{stage}: {err.splitlines()[0] if err else 'failed'}")

    return "\n".join(lines)


def build_chat_context(results):
    """Turns the current run's real CSVs into a text context for Gemini."""

    if not results:
        return None

    technique_df = load_csv_safe(results.get("technique_csv"))
    shot_df = load_csv_safe(results.get("shot_csv"))
    sync_df = load_csv_safe(results.get("sync_csv"))

    if technique_df.empty and shot_df.empty:
        return None

    parts = []

    summary = results.get("summary")
    if summary:
        parts.append(
            "OVERALL PERFORMANCE:\n"
            f"- Average Technique Score: {summary.get('average_score')}/100\n"
            f"- Total Shots Analyzed: {summary.get('total_shots')}\n"
            f"- Excellent: {summary.get('excellent')}\n"
            f"- Good: {summary.get('good')}\n"
            f"- Average: {summary.get('average')}\n"
            f"- Needs Improvement: {summary.get('needs_improvement')}"
        )

    if not technique_df.empty:
        parts.append("PER-SHOT TECHNIQUE EVALUATION:\n" + technique_df.to_string(index=False))

    if not sync_df.empty:
        parts.append("BALL-SHOT SYNCHRONIZATION (ball detection reliability per shot):\n" + sync_df.to_string(index=False))

    errors = results.get("errors") or {}
    if errors:
        parts.append(
            "PIPELINE NOTES (things that failed or were unavailable during analysis — "
            "be upfront about these if the user asks about them):\n"
            + "\n".join(f"- {k}: {v.splitlines()[0] if v else 'failed'}" for k, v in errors.items())
        )

    return "\n\n".join(parts)


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("uploaded_video", None)
st.session_state.setdefault("uploaded_video_name", None)
st.session_state.setdefault("uploaded_video_identity", None)
st.session_state.setdefault("results", None)
st.session_state.setdefault("chat_messages", [])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:20px">
        <div style="font-size:45px">🏏</div>
        <div style="font-size:27px;font-weight:800;">
            Cricket<span style="color:#00ffaa">AI</span>
        </div>
        <div style="color:#728797;font-size:10px;letter-spacing:2px;">
            BATTING INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🎥 Video Analysis",
            "📊 Technique Report",
            "🏏 Shot Analysis",
            "🤖 AI Cricket Coach",
        ],
    )

    st.divider()

    models_ok = os.path.exists(POSE_MODEL_PATH) and os.path.exists(BALL_MODEL_PATH)

    st.caption("Computer Vision")
    st.caption("Pose Estimation " + ("✅" if os.path.exists(POSE_MODEL_PATH) else "⚠️ model missing"))
    st.caption("Ball Detection " + ("✅" if os.path.exists(BALL_MODEL_PATH) else "⚠️ model missing"))
    st.caption("Gemini AI " + ("✅" if os.environ.get("GEMINI_API_KEY") else "⚠️ GEMINI_API_KEY not set"))


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">AI-POWERED CRICKET ANALYTICS</div>
        <h1>CricketAI</h1>
        <p>Turn a batting video into measurable technique, movement and shot insights.</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.markdown(
            '<div class="warn-box">⚠️ One or both YOLO model files '
            f'(<code>{os.path.basename(POSE_MODEL_PATH)}</code>, '
            f'<code>{os.path.basename(BALL_MODEL_PATH)}</code>) are not next to app.py. '
            'Analysis will fail until they\'re placed in the app folder.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">📤 Analyze a New Batting Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your cricket video here",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload a batting video for CricketAI analysis.",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # CHANGED (persistence fix): the file_uploader widget resets itself
    # whenever this branch of the script isn't rendered (i.e. any time
    # the user is on a different page). We only treat it as a genuinely
    # NEW upload when its identity actually changes — otherwise we fall
    # back to the path already saved in session_state, so switching
    # pages and coming back no longer loses the uploaded video.
    if uploaded is not None:

        upload_identity = getattr(uploaded, "file_id", None) or f"{uploaded.name}_{uploaded.size}"

        if st.session_state.get("uploaded_video_identity") != upload_identity:

            ext = os.path.splitext(uploaded.name)[1] or ".mp4"
            safe_name = f"{uuid.uuid4().hex[:10]}{ext}"
            uploaded_path = os.path.join(UPLOAD_DIR, safe_name)

            with open(uploaded_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.session_state.uploaded_video = uploaded_path
            st.session_state.uploaded_video_name = uploaded.name
            st.session_state.uploaded_video_identity = upload_identity
            st.session_state.results = None
            st.session_state.chat_messages = []

    current_video = st.session_state.uploaded_video

    if current_video and os.path.exists(current_video):

        display_name = st.session_state.get("uploaded_video_name", os.path.basename(current_video))
        st.success(f"Video ready: {display_name}")
        st.video(current_video)

        analyze_clicked = st.button(
            "▶ ANALYZE VIDEO",
            type="primary",
            disabled=not models_ok,
            use_container_width=True,
        )

        if analyze_clicked:

            progress_box = st.status("CRICKETAI ANALYSIS", expanded=True)
            progress_bar = st.progress(0.0)

            def on_progress(stage_index, stage_label, fraction):
                overall = (stage_index + (fraction if fraction is not None else 0)) / len(STAGE_LABELS)
                progress_bar.progress(min(overall, 1.0))
                progress_box.update(label=f"CRICKETAI ANALYSIS — [{stage_index + 1}/{len(STAGE_LABELS)}] {stage_label}")

            try:
                work_dir = new_work_dir(RUNS_DIR)

                results = run_analysis(
                    video_path=current_video,
                    work_dir=work_dir,
                    pose_model_path=POSE_MODEL_PATH,
                    ball_model_path=BALL_MODEL_PATH,
                    status_callback=on_progress,
                )

                st.session_state.results = results
                st.session_state.chat_messages = []  # fresh coach context for the new video

                progress_bar.progress(1.0)

                if results["errors"]:
                    progress_box.update(label="Analysis finished with some stages skipped", state="complete")
                    with progress_box:
                        st.warning("Some pipeline stages did not complete:")
                        for stage, err in results["errors"].items():
                            st.code(f"{stage}: {err.splitlines()[0]}", language=None)
                else:
                    progress_box.update(label="Analysis complete", state="complete")

                st.rerun()

            except Exception as e:
                progress_box.update(label="Analysis failed", state="error")
                st.error(f"Analysis failed: {e}")
                st.code(traceback.format_exc(), language=None)

    # --------------------------------------------------------
    # CURRENT PERFORMANCE
    # --------------------------------------------------------

    st.markdown('<div class="section">📊 Current Performance</div>', unsafe_allow_html=True)

    results = st.session_state.results
    summary = results.get("summary") if results else None

    if not summary:
        st.info("No analysis yet. Upload a video above and click Analyze to generate a real report.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            score_display = summary["average_score"] if summary["average_score"] is not None else "N/A"
            st.markdown(f"""
            <div class="metric">
                <div class="metric-title">Technique Score</div>
                <div class="metric-value">{score_display}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-title">Shots Analyzed</div>
                <div class="metric-value">{summary['total_shots']}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-title">Excellent</div>
                <div class="metric-value">{summary['excellent']}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-title">Needs Improvement</div>
                <div class="metric-value">{summary['needs_improvement']}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# VIDEO ANALYSIS
# ============================================================

elif page == "🎥 Video Analysis":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">VISUAL COMPUTER VISION</div>
        <h1>🎥 AI Motion Analysis</h1>
        <p>Player pose, ball detection, and shot data overlaid on your uploaded video.</p>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.results

    final_video = results.get("final_video") if results else None

    if final_video and os.path.exists(final_video):

        if results.get("video_web_compatible") is False:
            st.markdown(
                '<div class="warn-box">⚠️ This video could not be converted to a browser-friendly '
                'format, so the preview below may not play here — download it instead and it will '
                'open normally in VLC or any desktop player.</div>',
                unsafe_allow_html=True,
            )

        st.video(final_video)
        st.success("This is the generated analysis video — not the raw upload.")

        with open(final_video, "rb") as f:
            st.download_button(
                "⬇ Download Analyzed Video",
                data=f.read(),
                file_name="cricketai_analyzed_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    elif results:
        st.warning(
            "The final annotated video wasn't generated for this run. "
            f"Reason: {results['errors'].get('final_video', 'unknown error')}"
        )
    else:
        st.info("Upload and analyze a video from the Dashboard first.")


# ============================================================
# TECHNIQUE REPORT
# ============================================================

elif page == "📊 Technique Report":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">TECHNIQUE INTELLIGENCE</div>
        <h1>📊 Performance Report</h1>
        <p>Shot-by-shot evaluation of the detected batting technique.</p>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.results
    technique_df = load_csv_safe(results.get("technique_csv")) if results else pd.DataFrame()

    if technique_df.empty:
        st.warning("No technique report available yet. Analyze a video from the Dashboard first.")
    else:
        summary = results["summary"]

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                "⬇ Download Report (Text)",
                data=build_text_report(results),
                file_name="cricketai_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl_col2:
            with open(results["technique_csv"], "rb") as f:
                st.download_button(
                    "⬇ Download Shot Data (CSV)",
                    data=f.read(),
                    file_name="cricketai_technique_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        left, right = st.columns([1, 2])

        with left:
            score_display = f"{summary['average_score']}/100" if summary["average_score"] is not None else "N/A"
            st.metric("Overall Technique Score", score_display)

            missing_scores = technique_df["technique_score"].isna().sum()
            if missing_scores:
                st.caption(f"⚠️ {missing_scores} shot(s) have no reliable score — insufficient elbow/knee/timing data.")

        with right:
            categories = pd.DataFrame({
                "Category": ["Excellent", "Good", "Average", "Needs Improvement"],
                "Shots": [summary["excellent"], summary["good"], summary["average"], summary["needs_improvement"]],
            })
            st.bar_chart(categories.set_index("Category"))

        # ---------------- BEST SHOT HIGHLIGHT ----------------

        shot_images = results.get("shot_images") or {}
        scored_for_best = technique_df.dropna(subset=["technique_score"])

        if not scored_for_best.empty:

            best_row = scored_for_best.loc[scored_for_best["technique_score"].idxmax()]
            best_shot_number = int(best_row["shot_number"])
            best_image_path = shot_images.get(best_shot_number) or shot_images.get(str(best_shot_number))

            st.markdown('<div class="section">🏆 Best Shot</div>', unsafe_allow_html=True)

            img_col, info_col = st.columns([1.3, 1])

            with img_col:
                if best_image_path and os.path.exists(best_image_path):
                    st.image(best_image_path, use_container_width=True,
                              caption=f"Shot #{best_shot_number} — pose, ball and technique overlay")
                else:
                    st.info("No highlight image was captured for this shot (frame could not be matched exactly).")

            with info_col:
                st.markdown(f"""
                <div class="card">
                    <h3 style="margin-top:0">Shot #{best_shot_number} {badge(best_row['overall'])}</h3>
                    <div style="font-size:34px;font-weight:800;">{best_row['technique_score']:.1f}<span style="font-size:14px;color:#8096a7"> /100</span></div>
                    <p style="color:#a9bccb;margin-top:14px;">
                        Elbow: <b>{best_row['elbow']}</b><br>
                        Knee: <b>{best_row['knee']}</b><br>
                        Timing: <b>{best_row['timing']}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section">Shot Evaluation</div>', unsafe_allow_html=True)

        display_df = technique_df[[
            "shot_number", "technique_score", "overall", "elbow_angle", "elbow",
            "knee_angle", "knee", "timing", "ball_confidence",
        ]].copy()
        display_df.columns = [
            "Shot", "Score", "Overall", "Elbow °", "Elbow", "Knee °", "Knee", "Timing", "Ball Conf.",
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if "N/A" in technique_df["overall"].values or technique_df["timing"].eq("UNAVAILABLE").any():
            st.caption("⚠️ Some values above are N/A / UNAVAILABLE — this reflects missing pose or ball data, not a hidden failure.")

        score_chart = technique_df[["shot_number", "technique_score"]].dropna()
        if not score_chart.empty:
            st.markdown('<div class="section">Technique Score by Shot</div>', unsafe_allow_html=True)
            st.line_chart(score_chart.set_index("shot_number"))

        angle_chart = technique_df[["shot_number", "elbow_angle", "knee_angle"]].dropna(how="all", subset=["elbow_angle", "knee_angle"])
        if not angle_chart.empty:
            st.markdown('<div class="section">Elbow / Knee Angle by Shot</div>', unsafe_allow_html=True)
            st.line_chart(angle_chart.set_index("shot_number"))

        conf_chart = technique_df[["shot_number", "ball_confidence"]].dropna()
        if not conf_chart.empty:
            st.markdown('<div class="section">Ball Detection Confidence by Shot</div>', unsafe_allow_html=True)
            st.bar_chart(conf_chart.set_index("shot_number"))

        # ---------------- AI INSIGHTS ----------------

        st.markdown('<div class="section">🔍 AI Insights</div>', unsafe_allow_html=True)

        scored = technique_df.dropna(subset=["technique_score"])

        if scored.empty:
            st.info("Not enough reliable data across shots to generate insights.")
        else:
            best = scored.loc[scored["technique_score"].idxmax()]
            worst = scored.loc[scored["technique_score"].idxmin()]

            insights = [
                f"**Strongest shot:** #{int(best['shot_number'])} — {best['technique_score']:.1f}/100 ({best['overall']}).",
                f"**Weakest shot:** #{int(worst['shot_number'])} — {worst['technique_score']:.1f}/100 ({worst['overall']}).",
            ]

            closed_count = (technique_df["elbow"] == "TOO CLOSED").sum()
            straight_count = (technique_df["elbow"] == "TOO STRAIGHT").sum()
            unstable_count = (technique_df["knee"] == "UNSTABLE").sum()
            timing_issues = (technique_df["timing"] == "NEEDS IMPROVEMENT").sum()
            no_ball = (technique_df["timing"] == "UNAVAILABLE").sum()

            if closed_count:
                insights.append(f"Elbow was too closed on {closed_count} shot(s) — a repeated pattern worth drilling.")
            if straight_count:
                insights.append(f"Elbow was too straight/locked on {straight_count} shot(s).")
            if unstable_count:
                insights.append(f"Lower-body instability detected on {unstable_count} shot(s).")
            if timing_issues:
                insights.append(f"Ball timing needs work on {timing_issues} shot(s).")
            if no_ball:
                insights.append(f"⚠️ Ball detection was unreliable/missing for {no_ball} shot(s) — timing couldn't be evaluated for those.")

            for line in insights:
                st.markdown(f"- {line}")


# ============================================================
# SHOT ANALYSIS
# ============================================================

elif page == "🏏 Shot Analysis":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">SHOT INTELLIGENCE</div>
        <h1>🏏 Shot-by-Shot Analysis</h1>
        <p>Explore individual batting moments detected by CricketAI.</p>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.results
    technique_df = load_csv_safe(results.get("technique_csv")) if results else pd.DataFrame()

    if technique_df.empty:
        st.warning("No shot data available yet. Analyze a video from the Dashboard first.")
    else:
        shots = technique_df["shot_number"].dropna().tolist()
        selected_shot = st.selectbox("Select a shot", shots, format_func=lambda s: f"Shot #{int(s)}")

        row = technique_df[technique_df["shot_number"] == selected_shot]

        if not row.empty:
            row = row.iloc[0]

            score_text = f"{row['technique_score']:.1f}" if pd.notna(row["technique_score"]) else "N/A"

            shot_images = results.get("shot_images") or {}
            shot_num = int(row["shot_number"])
            image_path = shot_images.get(shot_num) or shot_images.get(str(shot_num))

            if image_path and os.path.exists(image_path):
                st.image(image_path, use_container_width=True,
                          caption=f"Shot #{shot_num} — pose, ball and technique overlay")

            st.markdown(f"""
            <div class="card">
                <h2 style="margin-top:0">SHOT #{int(row['shot_number'])} {badge(row['overall'])}</h2>
                <div style="font-size:38px;font-weight:800;">{score_text}<span style="font-size:16px;color:#8096a7"> /100</span></div>
            </div>
            """, unsafe_allow_html=True)

            a, b, c, d = st.columns(4)

            with a:
                st.metric("Elbow", row["elbow"] if pd.notna(row["elbow"]) else "N/A",
                           f"{row['elbow_angle']:.1f}°" if pd.notna(row["elbow_angle"]) else None)
            with b:
                st.metric("Knee", row["knee"] if pd.notna(row["knee"]) else "N/A",
                           f"{row['knee_angle']:.1f}°" if pd.notna(row["knee_angle"]) else None)
            with c:
                st.metric("Timing", row["timing"] if pd.notna(row["timing"]) else "N/A")
            with d:
                conf = row.get("ball_confidence")
                st.metric("Ball Confidence", f"{conf:.2f}" if pd.notna(conf) else "N/A")

            if pd.notna(row.get("advice")):
                st.markdown(f"""
                <div class="card">
                    <h3>💡 Coaching Insight</h3>
                    <p>{row['advice']}</p>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# GEMINI AI COACH
# ============================================================

elif page == "🤖 AI Cricket Coach":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">GEMINI-POWERED CRICKET INTELLIGENCE</div>
        <h1>🤖 AI Cricket Coach</h1>
        <p>Ask questions about your actual CricketAI analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    if not os.environ.get("GEMINI_API_KEY"):
        st.markdown(
            '<div class="warn-box">⚠️ <code>GEMINI_API_KEY</code> is not set as an environment variable. '
            'Set it before launching Streamlit (see setup notes) — the coach can\'t answer without it.</div>',
            unsafe_allow_html=True,
        )

    try:
        from chatbot import ask_cricket_coach
    except Exception as e:
        st.error(f"Unable to load chatbot.py: {e}")
        st.stop()

    results = st.session_state.results
    context = build_chat_context(results)

    if context is None:
        st.info("No analysis available yet — the coach will say so if you ask it something specific. Analyze a video from the Dashboard for grounded answers.")

    st.markdown("**Try asking:**")
    suggestions = [
        "What was my best shot?",
        "Why was my worst shot weak?",
        "What's my biggest weakness?",
        "How can I improve my elbow position?",
        "How can I improve my timing?",
        "What does my knee analysis mean?",
    ]

    cols = st.columns(3)
    clicked_suggestion = None
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, use_container_width=True):
            clicked_suggestion = s

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask your AI cricket coach...") or clicked_suggestion

    if question:

        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("CricketAI is analyzing..."):
                try:
                    answer = ask_cricket_coach(question, context)
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Gemini error: {e}")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<br><br>
<div style="text-align:center;padding:30px;border-top:1px solid rgba(255,255,255,.06);color:#536979;font-size:12px;">
    <b style="color:#8ca2b2;">CRICKET<span style="color:#00ffaa;">AI</span></b>
    <br><br>
    Computer Vision • Pose Estimation • Ball Detection • Technique Analytics • Gemini AI
</div>
""", unsafe_allow_html=True)
