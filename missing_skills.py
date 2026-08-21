import pandas as pd

# Load dataset
df = pd.read_csv("dataset/preprocessed_resume_data.csv")

# Replace empty values
df = df.fillna("")

# Read resume text
with open("resume_text.txt", "r", encoding="utf-8") as file:
    resume_text = file.read().lower()


# Skills present in the resume
resume_skills = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mongodb",
    "mysql",
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
    "computer vision"
]


# Find skills actually present in resume
found_skills = []

for skill in resume_skills:
    if skill in resume_text:
        found_skills.append(skill)


# Select the top recommended job
recommended_job = "ai engineer"


# Find rows related to the recommended job
job_rows = df[
    df["job_position_name"].str.lower() == recommended_job
]


# Collect required skills
required_skills = set()

for skills in job_rows["skills_required"]:

    skills = str(skills).lower()

    for skill in resume_skills:

        if skill in skills:
            required_skills.add(skill)


# Find missing skills
missing_skills = []

for skill in required_skills:

    if skill not in found_skills:
        missing_skills.append(skill)


# Display results
print("\n===== RECOMMENDED JOB =====")
print(recommended_job)

print("\n===== YOUR SKILLS =====")

for skill in sorted(found_skills):
    print("-", skill)

print("\n===== MISSING SKILLS =====")

if len(missing_skills) == 0:

    print("No major missing skills found!")

else:

    for skill in sorted(missing_skills):
        print("-", skill)

print("\nTotal Missing Skills:", len(missing_skills))