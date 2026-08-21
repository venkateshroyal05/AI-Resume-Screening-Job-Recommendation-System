import pandas as pd

df = pd.read_csv("dataset/resume_data_for_ranking.csv")

important_columns = [
    "skills",
    "career_objective",
    "degree_names",
    "major_field_of_studies",
    "positions",
    "job_position_name",
    "skills_required",
    "responsibilities.1",
    "matched_score"
]

df = df[important_columns]

df.to_csv("dataset/cleaned_resume_data.csv", index=False)

print("Cleaned dataset created successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))