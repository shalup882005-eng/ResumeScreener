"""
Resume Screener / ATS Score Checker
------------------------------------
Upload a resume (PDF) and paste a job description.
The app computes a match score using TF-IDF + cosine similarity,
and shows which key skills from the JD are missing in the resume.

Author: (your name)
"""

import re
import string

import streamlit as st
import pdfplumber
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# 1. A predefined skill list used to check "missing keywords".
#    Feel free to expand this list with skills relevant to your domain.
# ---------------------------------------------------------------------
SKILL_LIST = [
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "nosql",
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
    "streamlit", "fastapi", "machine learning", "deep learning", "nlp",
    "computer vision", "data analysis", "data visualization", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "opencv",
    "power bi", "tableau", "excel", "git", "github", "docker", "kubernetes",
    "aws", "azure", "gcp", "linux", "rest api", "api", "mongodb", "mysql",
    "postgresql", "hadoop", "spark", "agile", "scrum", "communication",
    "problem solving", "leadership", "teamwork", "oop", "dbms",
    "operating systems", "computer networks", "data structures",
    "algorithms", "statistics", "probability",
]


# ---------------------------------------------------------------------
# 2. Text extraction from PDF
# ---------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file object."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ---------------------------------------------------------------------
# 3. Text cleaning
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers/extra whitespace."""
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------
# 4. Similarity score using TF-IDF + cosine similarity
# ---------------------------------------------------------------------
def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Returns a similarity score between 0 and 100.
    TF-IDF turns each document into a vector where each word's weight
    reflects how important/rare it is. Cosine similarity then measures
    the angle between the two vectors — closer to 1 means more similar.
    """
    documents = [resume_text, jd_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)


# ---------------------------------------------------------------------
# 5. Skill matching — which JD skills are missing from the resume
# ---------------------------------------------------------------------
def find_skills(text: str, skill_list) -> set:
    text_lower = text.lower()
    found = set()
    for skill in skill_list:
        # word-boundary-ish match so "r" doesn't match inside "your"
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


# ---------------------------------------------------------------------
# 6. Streamlit UI
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="centered")

    st.title("📄 Resume Screener / ATS Score Checker")
    st.write(
        "Upload your resume (PDF) and paste a job description. "
        "This tool computes a match score and highlights skills you might be missing."
    )

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    with col2:
        jd_text_input = st.text_area("Paste Job Description", height=220)

    if st.button("Check Match", type="primary"):
        if resume_file is None or not jd_text_input.strip():
            st.warning("Please upload a resume and paste a job description first.")
            return

        with st.spinner("Analyzing..."):
            raw_resume_text = extract_text_from_pdf(resume_file)

            if not raw_resume_text.strip():
                st.error(
                    "Couldn't extract text from this PDF. "
                    "It might be a scanned/image-based resume."
                )
                return

            cleaned_resume = clean_text(raw_resume_text)
            cleaned_jd = clean_text(jd_text_input)

            score = compute_match_score(cleaned_resume, cleaned_jd)

            resume_skills = find_skills(raw_resume_text, SKILL_LIST)
            jd_skills = find_skills(jd_text_input, SKILL_LIST)
            missing_skills = jd_skills - resume_skills
            matched_skills = jd_skills & resume_skills

        # ---- Results ----
        st.subheader("Results")

        st.metric(label="Match Score", value=f"{score}%")
        st.progress(min(int(score), 100))

        if score >= 70:
            st.success("Strong match! Your resume aligns well with this job description.")
        elif score >= 40:
            st.info("Moderate match. Consider adding more relevant keywords.")
        else:
            st.warning("Low match. Your resume may need significant tailoring for this role.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Matched Skills**")
            if matched_skills:
                st.write(", ".join(sorted(matched_skills)))
            else:
                st.write("None found.")
        with c2:
            st.markdown("**❌ Missing Skills (present in JD, not in resume)**")
            if missing_skills:
                st.write(", ".join(sorted(missing_skills)))
            else:
                st.write("Great — no obvious skill gaps found!")

        # ---- Simple bar chart ----
        if jd_skills:
            fig, ax = plt.subplots(figsize=(6, 3))
            labels = ["Matched", "Missing"]
            values = [len(matched_skills), len(missing_skills)]
            ax.bar(labels, values, color=["#2ecc71", "#e74c3c"])
            ax.set_ylabel("Number of skills")
            ax.set_title("Skill Match Overview")
            st.pyplot(fig)

        with st.expander("How is this score calculated?"):
            st.write(
                "1. Text is extracted from your PDF resume and cleaned "
                "(lowercased, punctuation/numbers removed).\n"
                "2. Both the resume and job description are converted into "
                "TF-IDF vectors, which weigh words by how important/rare "
                "they are across the two documents.\n"
                "3. Cosine similarity is computed between the two vectors "
                "to produce a 0-100% match score.\n"
                "4. Separately, a predefined skill list is scanned in both "
                "texts to find skills present in the JD but missing from "
                "the resume."
            )


if __name__ == "__main__":
    main()
