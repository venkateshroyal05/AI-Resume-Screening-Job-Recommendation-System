from pypdf import PdfReader

pdf_path = "Venkatesh_Resume.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text()

print("Resume text extracted successfully!")
print(text)

with open("resume_text.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("Resume text saved successfully!")
print("\n--- EXTRACTED RESUME TEXT ---")
print(text[:3000])