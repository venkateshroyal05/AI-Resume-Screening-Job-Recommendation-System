import pandas as pd
import re

# Skills that the system can recognize
SKILLS = [
    "python",
    "java",
    "c++",
    "c",
    "sql",
    "mysql",
    "mongodb",
    "html",
    "css",
    "javascript",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "ai engineering",
    "tensorflow",
    "pytorch",
    "pandas",
    "numpy",
    "scikit learn",
    "scikit-learn",
    "statistics",
    "data visualization",
    "tableau",
    "power bi",
    "aws",
    "azure",
    "hadoop",
    "spark",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "nlp",
    "computer vision",
    "rest api",
    "software development",
    "business analysis",
    "effective communication",
    "design",
    "soft skills",
    "etl",
    "data analytics",
    "big data"
]


def contains_skill(text, skill):

    text = str(text).lower()

    if skill == "c":
        return bool(
            re.search(r"(?<![a-z+])c(?![a-z+])", text)
        )

    if skill == "c++":
        return "c++" in text

    if skill in ["scikit learn", "scikit-learn"]:
        return bool(
            re.search(r"scikit[- ]learn", text)
        )

    return bool(
        re.search(
            r"(?<![a-z])"
            + re.escape(skill)
            + r"(?![a-z])",
            text
        )
    )


def get_resume_skills(resume_text):

    found = []

    for skill in SKILLS:

        if contains_skill(resume_text, skill):
            found.append(skill)

    return set(found)


def get_job_skills(job_text):

    found = []

    for skill in SKILLS:

        if contains_skill(job_text, skill):
            found.append(skill)

    return set(found)


def find_missing_skills(
    resume_text,
    recommended_job
):

    df = pd.read_csv(
        "dataset/preprocessed_resume_data.csv"
    ).fillna("")

    job_rows = df[
        df["job_position_name"]
        .str.lower()
        == recommended_job.lower()
    ]

    required_skills = set()

    for value in job_rows["skills_required"]:

        required_skills.update(
            get_job_skills(value)
        )

    found_skills = get_resume_skills(
        resume_text
    )

    matched_skills = (
        required_skills
        & found_skills
    )

    missing_skills = (
        required_skills
        - found_skills
    )

    return (
        sorted(found_skills),
        sorted(matched_skills),
        sorted(missing_skills),
        sorted(required_skills)
    )