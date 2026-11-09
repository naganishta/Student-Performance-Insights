import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO
from datetime import datetime

# Optional ML dependency: install scikit-learn from requirements.txt
try:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="Student Insights | Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- DESIGN --------------------
st.markdown("""
<style>
.stApp {background:#F5F7FB;}
[data-testid="stSidebar"] {background:#101828;}
[data-testid="stSidebar"] * {color:#F2F4F7;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#101828;}
[data-testid="stSidebar"] input {color:#101828 !important;}
.block-container {padding-top:1.6rem; padding-bottom:3rem; max-width:1550px;}
.hero {background:linear-gradient(120deg,#101828,#1849A9);padding:28px 32px;
       border-radius:18px;margin-bottom:22px;color:white;}
.hero h1 {color:white;font-size:34px;margin:0;font-weight:750;}
.hero p {color:#D1E9FF;font-size:15px;margin:9px 0 0 0;}
.section {font-size:22px;font-weight:750;color:#101828;margin:20px 0 12px 0;}
div[data-testid="stMetric"] {background:white;border:1px solid #EAECF0;padding:16px;
border-radius:13px;box-shadow:0 2px 7px rgba(16,24,40,.04);}
div[data-testid="stMetricLabel"] {color:#475467;}
div[data-testid="stMetricValue"] {color:#175CD3;font-weight:750;}
div[data-testid="stPlotlyChart"] {background:white;border:1px solid #EAECF0;
border-radius:13px;padding:8px;}
.note {background:#EFF8FF;border-left:5px solid #2E90FA;padding:14px 18px;
border-radius:8px;color:#1849A9;margin:10px 0 18px 0;}
.footer {text-align:center;color:#667085;font-size:12px;padding-top:25px;}
</style>
""", unsafe_allow_html=True)

SCORE_COLS = ["math score", "reading score", "writing score"]
SUBJECT_LABELS = {"math score":"Math", "reading score":"Reading", "writing score":"Writing"}
CATEGORY_COLS = ["gender", "race/ethnicity", "parental level of education",
                 "lunch", "test preparation course"]
DEFAULT_FILE = Path(__file__).parent / "StudentsPerformance.csv"

# -------------------- DATA LOADING --------------------
st.sidebar.markdown("# 🎓 Student Insights")
st.sidebar.caption("STUDENT PERFORMANCE ANALYTICS")
st.sidebar.divider()
uploaded = st.sidebar.file_uploader("Upload a student CSV", type=["csv"])

@st.cache_data
def load_csv(file_bytes=None, file_name=None):
    if file_bytes is not None:
        return pd.read_csv(BytesIO(file_bytes))
    return pd.read_csv(file_name)

try:
    if uploaded is not None:
        raw_df = load_csv(uploaded.getvalue(), None)
        source_name = uploaded.name
    elif DEFAULT_FILE.exists():
        raw_df = load_csv(None, str(DEFAULT_FILE))
        source_name = DEFAULT_FILE.name
    else:
        st.title("Student Performance Analysis")
        st.info("Upload StudentsPerformance.csv from the sidebar to start.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

missing_required = [c for c in CATEGORY_COLS + SCORE_COLS if c not in raw_df.columns]
if missing_required:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing_required))
    st.stop()

df = raw_df.copy()
for col in SCORE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in CATEGORY_COLS:
    df[col] = df[col].astype("string").str.strip()

original_rows = len(df)
invalid_score_rows = int(df[SCORE_COLS].isna().any(axis=1).sum())
out_of_range_rows = int(((df[SCORE_COLS] < 0) | (df[SCORE_COLS] > 100)).any(axis=1).sum())
df = df.dropna(subset=SCORE_COLS + CATEGORY_COLS).copy()
df = df[~((df[SCORE_COLS] < 0) | (df[SCORE_COLS] > 100)).any(axis=1)].copy()
df = df.reset_index(drop=True)
df.insert(0, "Student Record", np.arange(1, len(df) + 1))
if df.empty:
    st.error("No valid student records remain after cleaning.")
    st.stop()

df["Average Score"] = df[SCORE_COLS].mean(axis=1).round(2)
df["Total Score"] = df[SCORE_COLS].sum(axis=1)
df["Score Range"] = df[SCORE_COLS].max(axis=1) - df[SCORE_COLS].min(axis=1)
df["Strongest Subject"] = df[SCORE_COLS].idxmax(axis=1).map(SUBJECT_LABELS)
df["Needs Most Support"] = df[SCORE_COLS].idxmin(axis=1).map(SUBJECT_LABELS)
df["Performance Level"] = pd.cut(
    df["Average Score"], bins=[-0.01, 39.99, 69.99, 100],
    labels=["Needs Support", "Developing", "High"]
).astype(str)

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.subheader("🔎 Dashboard Filters")
def multi_filter(label, column):
    options = sorted(df[column].dropna().astype(str).unique().tolist())
    return st.sidebar.multiselect(label, options, default=options)

gender_sel = multi_filter("Gender", "gender")
group_sel = multi_filter("Race / ethnicity", "race/ethnicity")
parent_sel = multi_filter("Parental education", "parental level of education")
lunch_sel = multi_filter("Lunch type", "lunch")
prep_sel = multi_filter("Test preparation", "test preparation course")
level_sel = st.sidebar.multiselect(
    "Performance level", ["Needs Support", "Developing", "High"],
    default=["Needs Support", "Developing", "High"]
)
score_range = st.sidebar.slider("Average score range", 0, 100, (0, 100))
filtered = df[
    df["gender"].isin(gender_sel)
    & df["race/ethnicity"].isin(group_sel)
    & df["parental level of education"].isin(parent_sel)
    & df["lunch"].isin(lunch_sel)
    & df["test preparation course"].isin(prep_sel)
    & df["Performance Level"].isin(level_sel)
    & df["Average Score"].between(score_range[0], score_range[1])
].copy()
st.sidebar.divider()
st.sidebar.caption(f"Source file: {source_name}")
st.sidebar.caption(f"Valid records: {len(df):,} / {original_rows:,}")

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
<h1>🎓 Student Performance Analytics</h1>
<p>Explore academic outcomes, compare student groups, inspect individual records,
and experiment with a score-estimation model.</p>
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    st.warning("No students match these filters. Adjust the sidebar selections.")
    st.stop()

# -------------------- TABS --------------------
tabs = st.tabs([
    "📊 Overview", "🧑‍🎓 Student Explorer", "⚖️ Group Comparison",
    "🔗 Relationships", "🧪 What-if Lab", "🤖 Score Estimator",
    "📁 Data & Reports"
])

# -------------------- OVERVIEW --------------------
with tabs[0]:
    st.markdown('<div class="section">Performance Overview</div>', unsafe_allow_html=True)
    avg_math = filtered["math score"].mean()
    avg_read = filtered["reading score"].mean()
    avg_write = filtered["writing score"].mean()
    avg_all = filtered["Average Score"].mean()
    pass_rate = (filtered["Average Score"] >= 40).mean() * 100
    high_rate = (filtered["Performance Level"] == "High").mean() * 100
    a,b,c,d,e = st.columns(5)
    a.metric("Students", f"{len(filtered):,}")
    b.metric("Average score", f"{avg_all:.1f}/100")
    c.metric("Math average", f"{avg_math:.1f}")
    d.metric("Reading average", f"{avg_read:.1f}")
    e.metric("Writing average", f"{avg_write:.1f}")
    f,g = st.columns(2)
    f.metric("At least 40 average", f"{pass_rate:.1f}%")
    g.metric("High performers (70+)", f"{high_rate:.1f}%")

    subject_summary = filtered[SCORE_COLS].mean().rename(index=SUBJECT_LABELS).rename_axis("Subject").reset_index(name="Average")
    left,right = st.columns(2)
    with left:
        fig = px.bar(subject_summary, x="Subject", y="Average", color="Subject",
                     text_auto=".1f", title="Average Score by Subject",
                     color_discrete_sequence=["#2E90FA","#12B76A","#7A5AF8"])
        fig.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0,100], height=360)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(filtered, x="Average Score", nbins=20,
                           color="Performance Level", title="Distribution of Student Averages",
                           color_discrete_sequence=["#F79009","#2E90FA","#12B76A"])
        fig.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(filtered[SCORE_COLS].rename(columns=SUBJECT_LABELS).melt(
        var_name="Subject", value_name="Score"), x="Subject", y="Score", color="Subject",
        title="Score Spread Across Subjects")
    fig.update_layout(template="plotly_white", showlegend=False, height=380, yaxis_range=[0,100])
    st.plotly_chart(fig, use_container_width=True)

    best_subject = subject_summary.loc[subject_summary["Average"].idxmax()]
    weakest_subject = subject_summary.loc[subject_summary["Average"].idxmin()]
    st.markdown(
        f'<div class="note"><b>Automatic insight:</b> {best_subject["Subject"]} has the highest '
        f'average ({best_subject["Average"]:.1f}), while {weakest_subject["Subject"]} has the '
        f'lowest average ({weakest_subject["Average"]:.1f}) in the currently filtered group. '
        'These are descriptive patterns, not evidence of causation.</div>',
        unsafe_allow_html=True
    )
    level_counts = filtered["Performance Level"].value_counts().reindex(
        ["Needs Support","Developing","High"], fill_value=0
    ).rename_axis("Level").reset_index(name="Students")
    fig = px.pie(level_counts, names="Level", values="Students", hole=.48,
                 title="Performance Level Breakdown")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- STUDENT EXPLORER --------------------
with tabs[1]:
    st.markdown('<div class="section">Individual Student Explorer</div>', unsafe_allow_html=True)
    st.caption("The original dataset has no student names or IDs. Record numbers are assigned for this dashboard only.")
    record_ids = filtered["Student Record"].tolist()
    chosen_id = st.selectbox("Choose a student record", record_ids)
    student = filtered.loc[filtered["Student Record"] == chosen_id].iloc[0]
    st.subheader(f"Student Record #{chosen_id}")
    x1,x2,x3,x4 = st.columns(4)
    x1.metric("Average", f'{student["Average Score"]:.1f}/100')
    x2.metric("Total", f'{student["Total Score"]:.0f}/300')
    x3.metric("Strongest subject", student["Strongest Subject"])
    x4.metric("Needs most support", student["Needs Most Support"])
    details = pd.DataFrame({
        "Attribute": ["Gender","Race / ethnicity","Parental education","Lunch","Test preparation","Performance level"],
        "Value": [student["gender"],student["race/ethnicity"],student["parental level of education"],
                  student["lunch"],student["test preparation course"],student["Performance Level"]]
    })
    st.dataframe(details, use_container_width=True, hide_index=True)
    scores = pd.DataFrame({"Subject":["Math","Reading","Writing"],
                           "Score":[student[c] for c in SCORE_COLS]})
    fig = px.bar(scores, x="Subject", y="Score", color="Subject", text_auto=True,
                 title="Selected Student: Subject Scores",
                 color_discrete_sequence=["#2E90FA","#12B76A","#7A5AF8"])
    fig.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0,100], height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download this student's record (CSV)",
                       student.to_frame().T.to_csv(index=False).encode("utf-8"),
                       file_name=f"student_record_{chosen_id}.csv", mime="text/csv")

    st.markdown("#### Top 10 students in the filtered group")
    top10 = filtered.nlargest(10, "Average Score")[
        ["Student Record"] + CATEGORY_COLS + SCORE_COLS + ["Average Score","Performance Level"]
    ]
    st.dataframe(top10, use_container_width=True, hide_index=True)

# -------------------- GROUP COMPARISON --------------------
with tabs[2]:
    st.markdown('<div class="section">Compare Student Groups</div>', unsafe_allow_html=True)
    group_by = st.selectbox("Compare groups by", CATEGORY_COLS,
                            index=CATEGORY_COLS.index("test preparation course"))
    metric = st.selectbox("Score to compare", SCORE_COLS + ["Average Score"],
                          format_func=lambda x: SUBJECT_LABELS.get(x,x))
    grouped = filtered.groupby(group_by, observed=False)[metric].agg(["mean","count","median"]).reset_index()
    grouped.columns = [group_by, "Average", "Students", "Median"]
    grouped = grouped.sort_values("Average", ascending=False)
    fig = px.bar(grouped, x=group_by, y="Average", color=group_by, text_auto=".1f",
                 title=f"{metric} by {group_by}")
    fig.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0,100], height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(grouped, use_container_width=True, hide_index=True)

    st.markdown("#### Compare all three subjects by group")
    subject_group = filtered.groupby(group_by, observed=False)[SCORE_COLS].mean().reset_index()
    melted = subject_group.melt(id_vars=group_by, var_name="Subject", value_name="Average")
    melted["Subject"] = melted["Subject"].map(SUBJECT_LABELS)
    fig = px.bar(melted, x=group_by, y="Average", color="Subject", barmode="group",
                 title=f"Subject averages across {group_by}")
    fig.update_layout(template="plotly_white", yaxis_range=[0,100], height=430)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- RELATIONSHIPS --------------------
with tabs[3]:
    st.markdown('<div class="section">Subject Relationships & Statistics</div>', unsafe_allow_html=True)
    corr = filtered[SCORE_COLS].corr()
    corr.index = [SUBJECT_LABELS[x] for x in corr.index]
    corr.columns = [SUBJECT_LABELS[x] for x in corr.columns]
    left,right = st.columns([1,1.3])
    with left:
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Subject Score Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        x_col = st.selectbox("X-axis subject", SCORE_COLS, index=0, format_func=lambda x: SUBJECT_LABELS[x])
        y_col = st.selectbox("Y-axis subject", SCORE_COLS, index=1, format_func=lambda x: SUBJECT_LABELS[x])
        fig = px.scatter(filtered, x=x_col, y=y_col, color="Performance Level",
                         hover_data=["Student Record","Average Score"],
                         trendline="ols", title=f"{SUBJECT_LABELS[x_col]} vs {SUBJECT_LABELS[y_col]}",
                         color_discrete_sequence=["#F79009","#2E90FA","#12B76A"])
        fig.update_layout(template="plotly_white", height=430)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Correlation values")
    st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

# -------------------- WHAT-IF LAB --------------------
with tabs[4]:
    st.markdown('<div class="section">What-if Score Lab</div>', unsafe_allow_html=True)
    st.write("Adjust hypothetical subject scores to see how the overall average changes. This is a calculator, not a prediction of real outcomes.")
    base_record = st.selectbox("Base this scenario on student record", filtered["Student Record"].tolist(), key="whatif_record")
    base_student = filtered.loc[filtered["Student Record"] == base_record].iloc[0]
    c1,c2,c3 = st.columns(3)
    with c1:
        new_math = st.slider("Math score", 0, 100, int(base_student["math score"]), key="wf_math")
    with c2:
        new_read = st.slider("Reading score", 0, 100, int(base_student["reading score"]), key="wf_read")
    with c3:
        new_write = st.slider("Writing score", 0, 100, int(base_student["writing score"]), key="wf_write")
    new_avg = np.mean([new_math,new_read,new_write])
    old_avg = base_student["Average Score"]
    delta = new_avg - old_avg
    q1,q2,q3 = st.columns(3)
    q1.metric("Scenario average", f"{new_avg:.2f}/100", f"{delta:+.2f} vs original")
    q2.metric("Original average", f"{old_avg:.2f}/100")
    q3.metric("Scenario total", f"{new_math+new_read+new_write}/300")
    comparison = pd.DataFrame({"Subject":["Math","Reading","Writing"],
                               "Original":[base_student[c] for c in SCORE_COLS],
                               "Scenario":[new_math,new_read,new_write]})
    fig = px.bar(comparison.melt(id_vars="Subject", var_name="Version", value_name="Score"),
                 x="Subject", y="Score", color="Version", barmode="group",
                 title="Original vs What-if Scores")
    fig.update_layout(template="plotly_white", yaxis_range=[0,100], height=380)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- SCORE ESTIMATOR --------------------
with tabs[5]:
    st.markdown('<div class="section">Machine Learning Score Estimator</div>', unsafe_allow_html=True)
    st.write("This optional Random Forest model estimates average score from the five categorical attributes. It is an educational demonstration, not a high-stakes student assessment.")
    if not SKLEARN_AVAILABLE:
        st.warning("Scikit-learn is not installed. Run: pip install scikit-learn")
    elif len(df) < 30:
        st.warning("At least 30 valid records are recommended to train this demonstration model.")
    else:
        feature_cols = CATEGORY_COLS
        X = df[feature_cols].astype(str)
        y = df["Average Score"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        preprocessor = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_cols)
        ])
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2))
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        m1,m2,m3 = st.columns(3)
        m1.metric("Mean absolute error", f"{mae:.2f} points")
        m2.metric("RMSE", f"{rmse:.2f} points")
        m3.metric("R² score", f"{r2:.3f}")
        st.caption("Metrics are calculated on a held-out test split. R² can be low or negative; that indicates the available attributes explain little of the score variation.")
        st.markdown("#### Estimate a hypothetical student's average")
        input_values = {}
        for col in CATEGORY_COLS:
            options = sorted(df[col].astype(str).unique().tolist())
            input_values[col] = st.selectbox(col.replace("_"," ").title(), options, key=f"predict_{col}")
        if st.button("Estimate average score", type="primary"):
            new_student = pd.DataFrame([input_values])
            estimate = float(model.predict(new_student)[0])
            st.success(f"Estimated average score: {estimate:.1f} / 100")
            st.caption("This is a statistical estimate based on patterns in this dataset, not a guaranteed result or a judgment about an individual.")
        # Permutation-free impurity feature importance is intentionally not displayed because
        # one-hot encoded category levels can make it easy to over-interpret model importance.

# -------------------- DATA & REPORTS --------------------
with tabs[6]:
    st.markdown('<div class="section">Data Quality, Records & Exports</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Original rows", f"{original_rows:,}")
    c2.metric("Valid rows", f"{len(df):,}")
    c3.metric("Rows with missing scores", f"{invalid_score_rows:,}")
    c4.metric("Rows with scores outside 0–100", f"{out_of_range_rows:,}")
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)
    st.markdown("#### Missing values by column")
    missing = raw_df.isna().sum().rename_axis("Column").reset_index(name="Missing values")
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df[SCORE_COLS + ["Average Score","Total Score"]].describe().T,
                 use_container_width=True)
    st.markdown("#### Export filtered records")
    export_cols = st.multiselect("Choose columns for CSV export", filtered.columns.tolist(),
                                 default=filtered.columns.tolist())
    export_df = filtered[export_cols] if export_cols else filtered[[]]
    st.download_button("⬇️ Download filtered data (CSV)",
                       export_df.to_csv(index=False).encode("utf-8"),
                       file_name="student_performance_filtered.csv", mime="text/csv")
    st.download_button("⬇️ Download full cleaned data (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="student_performance_cleaned.csv", mime="text/csv")

    # A simple PDF summary is generated with ReportLab if available.
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                                topMargin=18*mm, bottomMargin=18*mm)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Student Performance Analytics Report", styles["Title"]),
            Spacer(1, 8),
            Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles["Normal"]),
            Paragraph(f"Source file: {source_name}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph(f"Filtered students: {len(filtered)}", styles["Heading2"]),
        ]
        report_data = [["Metric","Value"],
                       ["Average math", f'{filtered["math score"].mean():.2f}'],
                       ["Average reading", f'{filtered["reading score"].mean():.2f}'],
                       ["Average writing", f'{filtered["writing score"].mean():.2f}'],
                       ["Overall average", f'{filtered["Average Score"].mean():.2f}'],
                       ["High performers", str(int((filtered["Performance Level"]=="High").sum()))]]
        table = Table(report_data, colWidths=[75*mm, 55*mm])
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1849A9")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#D0D5DD")),
            ("PADDING",(0,0),(-1,-1),7),
        ]))
        story.extend([table, Spacer(1,12),
                      Paragraph("Interpretation", styles["Heading2"]),
                      Paragraph("This report summarizes descriptive patterns in the selected records. "
                                "Group differences and correlations do not establish causation.", styles["BodyText"])])
        doc.build(story)
        st.download_button("📄 Download PDF summary report", buffer.getvalue(),
                           file_name="student_performance_report.pdf", mime="application/pdf")
    except ImportError:
        st.info("Install reportlab to enable PDF reports: pip install reportlab")

st.markdown('<div class="footer">Student Insights • Built with Python, Pandas, NumPy, Streamlit, Plotly and scikit-learn</div>',
            unsafe_allow_html=True)
