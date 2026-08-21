import pandas as pd
import re

# Load dataset
df = pd.read_csv("dataset/preprocessed_resume_data.csv")

# Replace empty values
df = df.fillna("")

# Read resume text
with open("resume_text.txt", "r", encoding="utf-8") as file:
    resume_text = file.read().lower()

# Skills that we want to check
common_skills = [
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

# Find skills in resume
matched_skills = []

for skill in common_skills:

    if skill in resume_text:
        matched_skills.append(skill)

# Display skills
print("\n===== SKILLS FOUND IN YOUR RESUME =====\n")

for skill in matched_skills:
    print("-", skill)

print("\nTotal Skills Found:", len(matched_skills))