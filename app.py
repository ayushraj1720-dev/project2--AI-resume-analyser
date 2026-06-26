# ============================================================
# AI Resume Analyzer
# Built with Python, NLP & Streamlit
# Tech: TF-IDF, Cosine Similarity, NLTK, PyPDF2, Matplotlib
# ============================================================

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ─────────────────────────────────────────
# Download NLTK Data (runs once, cached)
# ─────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng"]:
        nltk.download(pkg, quiet=True)

download_nltk()

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header ── */
.main-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* ── Score Card ── */
.score-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
}
.score-number {
    font-size: 3.8rem;
    font-weight: 700;
    line-height: 1;
}
.score-label {
    font-size: 1rem;
    opacity: 0.85;
    margin-top: 0.4rem;
}

/* ── Keyword Tags ── */
.tag-found {
    display: inline-block;
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem;
    margin: 0.2rem;
    font-weight: 500;
}
.tag-missing {
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffc107;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem;
    margin: 0.2rem;
    font-weight: 500;
}

/* ── Tip Cards ── */
.tip-card {
    background: #f0f4ff;
    border-left: 4px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    color: #333;
    line-height: 1.5;
}

/* ── Feature Cards (landing) ── */
.feature-card {
    background: white;
    border: 1px solid #e8e8e8;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    transition: transform 0.2s;
    height: 100%;
}
.feature-icon {
    font-size: 2.2rem;
    margin-bottom: 0.6rem;
}
.feature-title {
    font-weight: 600;
    font-size: 1rem;
    color: #333;
    margin-bottom: 0.4rem;
}
.feature-desc {
    color: #888;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ── Analyze Button ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.45) !important;
    transform: translateY(-1px);
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: #f8f9ff;
    border: 1px solid #dde2ff;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 AI Resume Analyzer")
    st.markdown("---")

    st.markdown("### 📌 How It Works")
    for step in [
        "1️⃣ Upload your resume (PDF)",
        "2️⃣ Paste the full job description",
        "3️⃣ Click **Analyze My Resume**",
        "4️⃣ Review your match score & gaps",
        "5️⃣ Improve and resubmit!",
    ]:
        st.markdown(step)

    st.markdown("---")
    st.markdown("### 🔬 Tech Stack")
    for tech, desc in [
        ("🐍 Python", "Core language"),
        ("🌊 Streamlit", "Web UI framework"),
        ("📊 Scikit-learn", "TF-IDF & Cosine Similarity"),
        ("📝 NLTK", "NLP text processing"),
        ("📄 PyPDF2", "PDF text extraction"),
        ("📈 Matplotlib", "Data visualization"),
    ]:
        st.markdown(f"**{tech}** — {desc}")

    st.markdown("---")
    st.info(
        "**💡 Quick Tips**\n\n"
        "✅ Mirror keywords from the JD\n\n"
        "✅ Quantify your achievements\n\n"
        "✅ List tools & technologies explicitly\n\n"
        "✅ Add a GitHub / portfolio link"
    )
    st.markdown("---")
    st.caption("Built with ❤️ | BTech CSE Project")


# ─────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────
st.markdown('<div class="main-title">🎯 AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Powered by NLP & Machine Learning &nbsp;•&nbsp; TF-IDF Cosine Similarity</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
# ─── HELPER FUNCTIONS ────────────────────
# ─────────────────────────────────────────

def extract_text_from_pdf(uploaded_file):
    """Extract all text from an uploaded PDF file."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def clean_text(text):
    """Lowercase text and strip non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    """Remove common English stopwords plus resume/JD filler words."""
    stop_words = set(stopwords.words("english"))
    extra = {
        "experience", "years", "year", "work", "working", "role", "position",
        "ability", "strong", "skills", "skill", "must", "will", "also",
        "good", "knowledge", "required", "requirements", "responsibilities",
        "including", "etc", "preferred", "plus", "bonus", "looking", "seeking",
    }
    stop_words.update(extra)
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in stop_words and len(w) > 2)


def extract_keywords(text, top_n=20):
    """
    Extract top-N meaningful keywords using POS tagging.
    Keeps nouns, verbs, and adjectives — filters out stopwords.
    """
    words = word_tokenize(text)
    words = [w.lower() for w in words if w.isalpha() and len(w) > 2]
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    tagged = pos_tag(words)
    meaningful = [w for w, pos in tagged if pos.startswith(("NN", "VB", "JJ"))]
    return Counter(meaningful).most_common(top_n)


def calculate_similarity(resume_text, job_text):
    """
    Compute TF-IDF cosine similarity between resume and job description.
    Uses bigrams (ngram_range=(1,2)) for better context capture.
    Returns: (score%, processed_resume, processed_jd)
    """
    resume_proc = remove_stopwords(clean_text(resume_text))
    job_proc    = remove_stopwords(clean_text(job_text))

    vectorizer   = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_proc, job_proc])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    return round(score, 2), resume_proc, job_proc


def get_keyword_gap(resume_text, job_text):
    """
    Find which job-description keywords appear in the resume
    and which ones are completely missing.
    """
    job_kws    = set(kw for kw, _ in extract_keywords(job_text, 20))
    resume_kws = set(kw for kw, _ in extract_keywords(resume_text, 40))

    found   = job_kws & resume_kws
    missing = job_kws - resume_kws
    return found, missing


def get_score_label(score):
    """Return (hex_color, emoji_label) based on score."""
    if score >= 70:
        return "#0f9d58", "🟢 Excellent Match"
    elif score >= 45:
        return "#f4b400", "🟡 Good Match"
    else:
        return "#db4437", "🔴 Low Match"


def draw_gauge(score):
    """Draw a semicircular gauge chart for the match score."""
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"projection": "polar"})

    import math
    # Background grey arc
    theta_bg = [i * math.pi / 180 for i in range(0, 181)]
    ax.plot(theta_bg, [1] * len(theta_bg), color="#e8e8e8", linewidth=22, solid_capstyle="round")

    # Coloured score arc
    color, _ = get_score_label(score)
    filled = int(score * 1.8)          # map 0-100 → 0-180 degrees
    theta_score = [i * math.pi / 180 for i in range(0, filled)]
    if theta_score:
        ax.plot(theta_score, [1] * len(theta_score),
                color=color, linewidth=22, solid_capstyle="round")

    # Centre score text
    ax.text(math.pi / 2, 0, f"{score:.0f}%",
            ha="center", va="center",
            fontsize=30, fontweight="bold", color="#333333")

    ax.set_ylim(0, 1.6)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.tight_layout(pad=0)
    return fig


def draw_bar_chart(keywords, title, bar_color):
    """Horizontal bar chart for keyword frequency."""
    if not keywords:
        return None

    words  = [kw  for kw,  _ in keywords[:10]]
    counts = [cnt for _,  cnt in keywords[:10]]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.barh(words[::-1], counts[::-1],
                   color=bar_color, alpha=0.82, edgecolor="white")
    ax.set_xlabel("Frequency", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)

    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=8, color="#555")

    fig.patch.set_facecolor("none")
    ax.set_facecolor("#fafafa")
    plt.tight_layout()
    return fig


def build_recommendations(score, missing_kws, resume_text):
    """Generate personalised, actionable resume improvement tips."""
    tips = []

    # Score-based opening
    if score < 40:
        tips.append("🔴 **Low Match Score** — Your resume needs significant tailoring for this specific role.")
        tips.append("📝 Rewrite your objective/summary section using language from the job description.")
    elif score < 70:
        tips.append("🟡 **Moderate Match** — A few targeted tweaks will make a big difference.")
        tips.append("📝 Add a dedicated **Skills** section listing the tools & technologies required in the JD.")
    else:
        tips.append("🟢 **Excellent Match!** — You look great for this role; just do a final polish.")
        tips.append("📝 Mirror the job title exactly in your resume headline and summary.")

    # Missing keywords tip
    if missing_kws:
        top5 = ", ".join(list(missing_kws)[:5])
        tips.append(f"🔑 **Add these missing keywords:** {top5}")

    # Length check
    if len(resume_text.split()) < 300:
        tips.append("📄 **Resume is too short** — aim for 400–600 words to cover all areas properly.")

    # Fresher-specific sections
    if "project" not in resume_text.lower():
        tips.append("💡 **Add a Projects section** — for freshers, projects are your most powerful proof of skills.")

    if "github" not in resume_text.lower():
        tips.append("🔗 **Add your GitHub link** — interviewers love being able to see your actual code.")

    if "intern" not in resume_text.lower():
        tips.append("🏢 **Include internship / training experience** even if it was short — it counts!")

    tips.append("📏 **Use bullet points**, not paragraphs — recruiters scan; they don't read full sentences.")
    tips.append("🔢 **Quantify achievements** — 'Improved model accuracy by 12%' beats 'Improved accuracy'.")
    tips.append("📐 **Keep resume to 1 page** as a fresher — concise is professional.")

    return tips


def build_text_report(score, score_label, found_kws, missing_kws, job_keywords,
                      resume_keywords, recommendations):
    """Build a plain-text download report."""
    sep = "=" * 55
    report = f"""AI RESUME ANALYZER — ANALYSIS REPORT
{sep}

MATCH SCORE : {score:.2f}%
STATUS      : {score_label}

{sep}
KEYWORDS FOUND IN YOUR RESUME ({len(found_kws)}):
{', '.join(sorted(found_kws)) if found_kws else 'None matched — consider adding job-specific terms.'}

KEYWORDS MISSING FROM YOUR RESUME ({len(missing_kws)}):
{', '.join(sorted(missing_kws)) if missing_kws else 'None! Great coverage.'}

{sep}
TOP JOB DESCRIPTION KEYWORDS:
{', '.join(kw for kw, _ in job_keywords)}

TOP RESUME KEYWORDS:
{', '.join(kw for kw, _ in resume_keywords)}

{sep}
PERSONALISED RECOMMENDATIONS:
"""
    for i, tip in enumerate(recommendations, 1):
        clean = re.sub(r"\*\*|[🔴🟡🟢🔑📝💡🔗📏🔢📄🏢📐]", "", tip).strip()
        report += f"  {i}. {clean}\n"

    report += f"\n{sep}\nGenerated by AI Resume Analyzer | BTech CSE Project\n"
    return report


# ─────────────────────────────────────────
# ─── INPUT SECTION ───────────────────────
# ─────────────────────────────────────────
st.markdown("")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📄 Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["pdf"],
        help="Only PDF format is supported",
    )
    if uploaded_file:
        st.success(f"✅ Uploaded: **{uploaded_file.name}**")

with col2:
    st.markdown("### 📋 Paste Job Description")
    job_description = st.text_area(
        "Copy & paste the full job description here",
        height=210,
        placeholder=(
            "Paste the full job description here...\n\n"
            "Example:\n"
            "We are looking for a Python Developer with 0–2 years of experience "
            "in Django, REST APIs, PostgreSQL, and Git. Familiarity with Machine "
            "Learning is a plus..."
        ),
    )
    if job_description:
        st.caption(f"📝 {len(job_description.split())} words entered")

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("🔍 Analyze My Resume", use_container_width=True)


# ─────────────────────────────────────────
# ─── ANALYSIS SECTION ────────────────────
# ─────────────────────────────────────────
if analyze:
    # ── Validation ──
    if not uploaded_file:
        st.warning("⚠️ Please upload your PDF resume first.")
        st.stop()
    if not job_description.strip():
        st.warning("⚠️ Please paste the job description.")
        st.stop()

    with st.spinner("🤖 Analyzing your resume with AI…"):

        resume_text = extract_text_from_pdf(uploaded_file)
        if not resume_text:
            st.error("❌ Could not extract text from this PDF. Try another file.")
            st.stop()

        # Core NLP processing
        score, resume_proc, job_proc = calculate_similarity(resume_text, job_description)
        resume_kws = extract_keywords(resume_proc, 15)
        job_kws    = extract_keywords(job_proc,    15)
        found_kws, missing_kws = get_keyword_gap(resume_proc, job_proc)
        _, score_label = get_score_label(score)
        recommendations = build_recommendations(score, missing_kws, resume_text)

    # ─── Results Header ───
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # ─── Score Row ───
    sc_col, gauge_col, stats_col = st.columns([1, 1.2, 1])

    with sc_col:
        _, lbl = get_score_label(score)
        st.markdown(f"""
        <div class="score-card">
            <div style="font-size:0.95rem; opacity:0.8; margin-bottom:0.4rem;">Your Match Score</div>
            <div class="score-number">{score:.1f}%</div>
            <div class="score-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    with gauge_col:
        st.pyplot(draw_gauge(score), use_container_width=True)

    with stats_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("📝 Resume Words",  len(resume_text.split()))
        st.metric("📋 JD Words",      len(job_description.split()))
        st.metric("🔑 JD Keywords",   len(job_kws))
        st.metric("✅ Keywords Found", len(found_kws))

    st.markdown("---")

    # ─── Keyword Gap Analysis ───
    st.markdown("### 🔑 Keyword Gap Analysis")
    st.caption("Keywords pulled from the job description — green = already in your resume, amber = missing.")

    kw_col1, kw_col2 = st.columns(2)

    with kw_col1:
        st.markdown("#### ✅ Found in Your Resume")
        if found_kws:
            html = "".join(
                f'<span class="tag-found">✓ {kw}</span>'
                for kw in sorted(found_kws)
            )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No matching keywords found — add job-specific terms to your resume.")

    with kw_col2:
        st.markdown("#### ⚠️ Missing from Your Resume")
        if missing_kws:
            html = "".join(
                f'<span class="tag-missing">✗ {kw}</span>'
                for kw in sorted(missing_kws)
            )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.success("🎉 No critical keywords missing — great job!")

    st.markdown("---")

    # ─── Keyword Frequency Charts ───
    st.markdown("### 📈 Keyword Frequency Charts")
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        fig1 = draw_bar_chart(job_kws, "🎯 Top Job Description Keywords", "#667eea")
        if fig1:
            st.pyplot(fig1, use_container_width=True)

    with ch_col2:
        fig2 = draw_bar_chart(resume_kws, "📄 Top Resume Keywords", "#764ba2")
        if fig2:
            st.pyplot(fig2, use_container_width=True)

    st.markdown("---")

    # ─── Recommendations ───
    st.markdown("### 💡 Personalised Recommendations")
    for tip in recommendations:
        st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ─── Resume Text Preview ───
    with st.expander("🔍 View Extracted Resume Text", expanded=False):
        st.text_area("Raw text extracted from your PDF:", resume_text,
                     height=300, disabled=True)

    # ─── Download Report ───
    st.markdown("### 📥 Download Analysis Report")
    report_text = build_text_report(
        score, score_label, found_kws, missing_kws,
        job_kws, resume_kws, recommendations
    )
    st.download_button(
        label="📄 Download Full Report (.txt)",
        data=report_text,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
    )

    st.markdown(
        '<div style="text-align:center;color:#aaa;font-size:0.82rem;margin-top:2rem;">'
        "⚡ Powered by TF-IDF Cosine Similarity & NLTK NLP &nbsp;•&nbsp; Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# ─── LANDING PAGE (before analysis) ──────
# ─────────────────────────────────────────
if not analyze:
    st.markdown("<br>", unsafe_allow_html=True)
    features = [
        ("📊", "Smart Scoring",
         "TF-IDF + Cosine Similarity gives you an objective, data-driven match score"),
        ("🔑", "Keyword Gap Analysis",
         "See exactly which keywords from the job description are missing from your resume"),
        ("📈", "Visual Charts",
         "Frequency charts for both your resume and the job description side-by-side"),
        ("💡", "Actionable Tips",
         "Personalised recommendations to improve your resume and boost your score"),
        ("🎓", "Fresher-Friendly",
         "Specific advice on projects, GitHub links, internships, and formatting"),
        ("📥", "Download Report",
         "Export your full analysis as a text report to keep or share"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)