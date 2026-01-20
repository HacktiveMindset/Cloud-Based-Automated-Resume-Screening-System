import os
import re
import pdfplumber
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load trained ML model and TF-IDF vectorizer
model = joblib.load("models/resume_classifier_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer_model.pkl")

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# Extract email
def extract_email(text):
    emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return emails[0] if emails else ""

# Extract phone (Improved regex for better detection of phone numbers)
def extract_phone(text):
    phones = re.findall(r'(?:(?:\+?\d{1,3})?[\s.-]?)?(?:\(?\d{3,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}', text)
    phones = [p.strip() for p in phones if len(p.strip()) >= 10]
    if phones:
        # Format to remove any unwanted characters like spaces or dashes
        return re.sub(r'\D', '', phones[0])  # Remove non-digit characters
    return ""

# Screen resumes based on job description
def screen_resumes(resume_texts, job_description):
    job_vec = vectorizer.transform([job_description])
    resume_vecs = vectorizer.transform(resume_texts)
    similarities = resume_vecs.dot(job_vec.T).toarray().flatten()
    return similarities

# Normalize match scores (Scale from 0 to 100, with the highest being 100)
def normalize_scores(scores):
    max_score = max(scores)
    return [(score / max_score) * 100 for score in scores]

# Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resumes = request.files.getlist('resumes')
        job_description = request.form.get('jd', '')
        results = []

        for resume in resumes:
            filename = secure_filename(resume.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume.save(filepath)

            text = extract_text_from_pdf(filepath)
            email = extract_email(text)
            phone = extract_phone(text)

            results.append({
                'file': filename,
                'text': text,
                'email': email,
                'phone': phone
            })

        # Matching scores
        match_scores = screen_resumes([r['text'] for r in results], job_description)

        # Normalize the match scores from 0 to 100, with the highest match being 100
        normalized_scores = normalize_scores(match_scores)

        # Assign the normalized scores to the results
        for i, score in enumerate(normalized_scores):
            results[i]['match'] = round(float(score), 2)

        # Sort by highest match
        results.sort(key=lambda x: x['match'], reverse=True)

        # Export CSV
        df = pd.DataFrame(results)
        df = df[['file', 'email', 'phone', 'match']]
        df.to_csv('matched_resumes.csv', index=False)

        return render_template('results_new.html', tables=df.to_html(index=False, classes='table table-striped table-bordered'))

    return render_template('index_new.html')

# CSV download
@app.route('/download')
def download():
    return send_file("matched_resumes.csv", as_attachment=True)

# Run server
if __name__ == '__main__':
    app.run(debug=True)
