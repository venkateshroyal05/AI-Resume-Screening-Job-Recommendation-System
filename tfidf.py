import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load job dataset
df = pd.read_csv("dataset/preprocessed_resume_data.csv")

# Replace empty values
df = df.fillna("")

# Create job text
df["job_text"] = (
    df["job_position_name"] + " " +
    df["skills_required"] + " " +
    df["responsibilities.1"]
)

# Read your resume
with open("resume_text.txt", "r", encoding="utf-8") as file:
    resume_text = file.read()

print("Resume loaded successfully!")

# Create TF-IDF
vectorizer = TfidfVectorizer()

# Convert job descriptions into numbers
job_vectors = vectorizer.fit_transform(df["job_text"])

# Convert your resume into numbers
resume_vector = vectorizer.transform([resume_text])

print("TF-IDF conversion completed!")

# Calculate similarity
similarity_scores = cosine_similarity(resume_vector, job_vectors)[0]

print("Cosine similarity calculation completed!")

# Add scores to dataset
df["similarity_score"] = similarity_scores

# Sort jobs by similarity
top_jobs = df.sort_values(
    by="similarity_score",
    ascending=False
).drop_duplicates(
    subset=["job_position_name"]
).head(5)

# Display Top 5 jobs
print("\nTop 5 Recommended Jobs:")

for index, row in top_jobs.iterrows():
    job_name = row["job_position_name"]
    score = row["similarity_score"] * 100

    print(f"{job_name} - {score:.2f}%")