# Resume Screener / ATS Score Checker

A simple NLP-based tool that compares a resume (PDF) against a job
description and returns a match score, along with a list of skills
present in the job description but missing from the resume.

## Problem Statement
Recruiters and ATS (Applicant Tracking Systems) often filter resumes by
keyword match before a human ever reads them. This tool lets a student
check, in advance, how well their resume aligns with a given job
description — and exactly which skills to add.

## Approach
1. **Text extraction** — `pdfplumber` extracts raw text from the uploaded
   PDF resume.
2. **Cleaning** — text is lowercased, punctuation/numbers stripped.
3. **Vectorization** — both resume and job description are converted
   into TF-IDF vectors using `scikit-learn`'s `TfidfVectorizer`. TF-IDF
   weighs each word by how important/rare it is across the two documents.
4. **Similarity** — cosine similarity between the two vectors gives a
   0-100% match score.
5. **Skill gap analysis** — a predefined list of ~60 common technical and
   soft skills is scanned in both texts; skills in the JD but absent from
   the resume are flagged as "missing."
6. **UI** — built with `Streamlit` for a simple, interactive web interface.

## Tech Stack
- Python
- Streamlit (UI)
- pdfplumber (PDF text extraction)
- scikit-learn (TF-IDF + cosine similarity)
- Matplotlib (skill match chart)

## How to Run Locally
```bash
# 1. Clone/download this folder, then cd into it
cd resume-screener

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) test the core logic without the UI
python test_logic.py

# 4. Run the app
streamlit run app.py
```
The app will open at `http://localhost:8501`.

## How to Deploy (free)
1. Push this folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub.
3. Click "New app," select your repo and `app.py` as the entry point.
4. Deploy — you'll get a public link you can put on your resume.

## Possible Improvements (good talking points in interviews)
- Replace TF-IDF with sentence embeddings (`sentence-transformers`) for
  semantic matching, e.g. recognizing "ML" ≈ "Machine Learning."
- Expand the skill list or make it dynamic (extract skills from the JD
  automatically using NER instead of a fixed list).
- Support DOCX resumes, not just PDF.
- Add resume formatting/ATS-friendliness checks (e.g., tables, images
  that ATS systems can't parse).

## Author
(Shalu Patel) — B.Tech CSE(AIML), GGITS Jabalpur
