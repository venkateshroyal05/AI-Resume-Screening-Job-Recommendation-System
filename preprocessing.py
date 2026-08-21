import pandas as pd
import re

df = pd.read_csv("dataset/cleaned_resume_data.csv")


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


text_columns = [
    "skills",
    "career_objective",
    "degree_names",
    "major_field_of_studies",
    "positions",
    "job_position_name",
    "skills_required",
    "responsibilities.1"
]

for column in text_columns:
    df[column] = df[column].apply(clean_text)


df.to_csv("dataset/preprocessed_resume_data.csv", index=False)

print("Preprocessing completed successfully!")
print("Rows:", len(df))