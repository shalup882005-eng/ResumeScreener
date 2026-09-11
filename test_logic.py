"""
Quick standalone test — run this FIRST before touching Streamlit,
to make sure the core matching logic works.

Run with: python test_logic.py
"""

from app import clean_text, compute_match_score, find_skills, SKILL_LIST

sample_resume = """
I am a final year B.Tech CSE student skilled in Python, Java, and SQL.
I have built projects using Pandas, NumPy, and scikit-learn for machine
learning. Familiar with Git, GitHub, and basic web development using
HTML and CSS. Strong problem solving and teamwork skills.
"""

sample_jd = """
We are looking for a Data Analyst with strong skills in Python, SQL,
Excel, and Power BI. Experience with Pandas, data visualization, and
statistics is required. Knowledge of machine learning and communication
skills are a plus.
"""

if __name__ == "__main__":
    cleaned_resume = clean_text(sample_resume)
    cleaned_jd = clean_text(sample_jd)

    score = compute_match_score(cleaned_resume, cleaned_jd)
    resume_skills = find_skills(sample_resume, SKILL_LIST)
    jd_skills = find_skills(sample_jd, SKILL_LIST)
    missing = jd_skills - resume_skills

    print(f"Match Score: {score}%")
    print(f"Skills found in resume: {sorted(resume_skills)}")
    print(f"Skills found in JD: {sorted(jd_skills)}")
    print(f"Missing skills: {sorted(missing)}")
