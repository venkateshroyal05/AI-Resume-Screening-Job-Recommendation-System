import streamlit as st
import pandas as pd

from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from missing_skills import find_missing_skills


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Resume Job Recommendation",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .job-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Resume Screening & Job Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload your resume to get AI-based job recommendations and skill analysis.</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = pd.read_csv(
        "dataset/preprocessed_resume_data.csv"
    )

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()


df = df.fillna("")


# ============================================================
# CREATE JOB TEXT
# ============================================================

df["job_text"] = (
    df["job_position_name"].astype(str)
    + " "
    + df["skills_required"].astype(str)
    + " "
    + df["responsibilities.1"].astype(str)
)


# ============================================================
# RESUME UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload your Resume (PDF)",
    type=["pdf"]
)


# ============================================================
# WHEN RESUME IS UPLOADED
# ============================================================

if uploaded_file is not None:

    st.success(
        "✅ Resume uploaded successfully!"
    )

    st.write(
        "**File name:**",
        uploaded_file.name
    )

    st.write("")

    analyze = st.button(
        "🔍 Analyze Resume",
        use_container_width=True
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if analyze:

        # ====================================================
        # EXTRACT RESUME TEXT
        # ====================================================

        try:

            reader = PdfReader(
                uploaded_file
            )

            resume_text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    resume_text += (
                        page_text + " "
                    )

            resume_text = resume_text.lower().strip()

        except Exception as e:

            st.error(
                f"Unable to read resume: {e}"
            )

            st.stop()


        # ====================================================
        # CHECK TEXT
        # ====================================================

        if not resume_text:

            st.error(
                "No readable text was found in the PDF."
            )

            st.stop()


        # ====================================================
        # EXTRACTED TEXT
        # ====================================================

        st.subheader(
            "📄 Extracted Resume Text"
        )

        st.text_area(
            "Resume Content",
            resume_text,
            height=250
        )


        # ====================================================
        # TF-IDF
        # ====================================================

        with st.spinner(
            "Calculating TF-IDF similarity..."
        ):

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            job_vectors = vectorizer.fit_transform(
                df["job_text"]
            )

            resume_vector = vectorizer.transform(
                [resume_text]
            )


        # ====================================================
        # COSINE SIMILARITY
        # ====================================================

        similarity_scores = cosine_similarity(
            resume_vector,
            job_vectors
        )[0]


        df["tfidf_score"] = (
            similarity_scores
        )


        # ====================================================
        # SELECT CANDIDATE JOBS
        # ====================================================

        candidates = (
            df
            .sort_values(
                by="tfidf_score",
                ascending=False
            )
            .drop_duplicates(
                subset=["job_position_name"]
            )
            .head(20)
        )


        results = []


        # ====================================================
        # CALCULATE JOB SCORES
        # ====================================================

        with st.spinner(
            "Analyzing skills and recommending jobs..."
        ):

            for _, row in candidates.iterrows():

                job_name = (
                    row["job_position_name"]
                )

                tfidf_score = float(
                    row["tfidf_score"]
                )


                # --------------------------------------------
                # FIND SKILLS
                # --------------------------------------------

                (
                    found_skills,
                    matched_skills,
                    missing_skills,
                    required_skills
                ) = find_missing_skills(
                    resume_text,
                    job_name
                )


                # --------------------------------------------
                # SKILL MATCH
                # --------------------------------------------

                if len(required_skills) > 0:

                    skill_score = (
                        len(matched_skills)
                        /
                        len(required_skills)
                    )

                else:

                    skill_score = 0.0


                # --------------------------------------------
                # FINAL SCORE
                # --------------------------------------------

                final_score = (
                    (skill_score * 70)
                    +
                    (tfidf_score * 30)
                )


                results.append({

                    "job": job_name,

                    "final": final_score,

                    "tfidf": tfidf_score,

                    "skill_score": skill_score,

                    "found": found_skills,

                    "matched": matched_skills,

                    "missing": missing_skills,

                    "required": required_skills

                })


        # ====================================================
        # SORT RESULTS
        # ====================================================

        results = sorted(
            results,
            key=lambda x: x["final"],
            reverse=True
        )


        # Top 5
        results = results[:5]


        # ====================================================
        # TOP 5
        # ====================================================

        st.subheader(
            "🎯 Top 5 Recommended Jobs"
        )


        # ====================================================
        # DISPLAY EACH JOB
        # ====================================================

        for number, result in enumerate(
            results,
            start=1
        ):

            job_title = (
                str(result["job"]).title()
            )


            st.markdown(
                f"## {number}. {job_title}"
            )


            # ------------------------------------------------
            # SCORE COLUMNS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "🎯 Final Match",
                    f"{result['final']:.2f}%"
                )


            with col2:

                st.metric(
                    "📊 TF-IDF Similarity",
                    f"{result['tfidf'] * 100:.2f}%"
                )


            with col3:

                st.metric(
                    "🧠 Skill Match",
                    f"{result['skill_score'] * 100:.2f}%"
                )


            # ------------------------------------------------
            # SCORE BAR
            # ------------------------------------------------

            st.progress(
                min(
                    max(
                        result["final"] / 100,
                        0
                    ),
                    1
                )
            )


            # =================================================
            # REQUIRED SKILLS
            # =================================================

            st.markdown(
                "### 📋 Required Skills"
            )


            if result["required"]:

                required_text = " • ".join(
                    skill.title()
                    for skill in result["required"]
                )

                st.info(
                    required_text
                )

            else:

                st.write(
                    "No recognized required skills."
                )


            # =================================================
            # MATCHED / MISSING
            # =================================================

            col1, col2 = st.columns(2)


            # -------------------------------------------------
            # MATCHED
            # -------------------------------------------------

            with col1:

                st.markdown(
                    "### ✅ Matched Skills"
                )


                if result["matched"]:

                    for skill in sorted(
                        result["matched"]
                    ):

                        st.success(
                            f"✓ {skill.title()}"
                        )

                else:

                    st.write(
                        "No matched skills found."
                    )


            # -------------------------------------------------
            # MISSING
            # -------------------------------------------------

            with col2:

                st.markdown(
                    "### ❌ Missing Skills"
                )


                if result["missing"]:

                    for skill in sorted(
                        result["missing"]
                    ):

                        st.warning(
                            f"• {skill.title()}"
                        )

                else:

                    st.success(
                        "No major missing skills! 🎉"
                    )


            st.divider()


        # ====================================================
        # RESUME SUMMARY
        # ====================================================

        st.subheader(
            "📊 Resume Analysis Summary"
        )


        # Get unique resume skills
        all_resume_skills = set()


        for result in results:

            all_resume_skills.update(
                result["found"]
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Recommended Jobs",
                len(results)
            )


        with col2:

            st.metric(
                "Skills Detected",
                len(all_resume_skills)
            )


        with col3:

            st.metric(
                "Analysis Method",
                "TF-IDF + Skills"
            )


        # ====================================================
        # DETECTED SKILLS
        # ====================================================

        st.markdown(
            "### 🧠 Skills Detected in Resume"
        )


        if all_resume_skills:

            skills_text = " • ".join(
                skill.title()
                for skill in sorted(
                    all_resume_skills
                )
            )

            st.success(
                skills_text
            )

        else:

            st.warning(
                "No recognized skills detected."
            )


        # ====================================================
        # FINAL MESSAGE
        # ====================================================

        st.success(
            "✅ Resume analysis completed successfully!"
        )


else:

    st.info(
        "📄 Please upload your resume to continue."
    )