import os
import re
import pdfplumber
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("models/resume_classifier_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer_model.pkl")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else ""

def extract_phone(text):
    match = re.search(r"(\+91[-\s]?)?[0]?(91)?[789]\d{9}", text)
    return match.group(0) if match else ""

def screen_resumes(resume_texts, job_description):
    job_vec = vectorizer.transform([job_description])
    resume_vecs = vectorizer.transform(resume_texts)
    similarities = resume_vecs.dot(job_vec.T).toarray().flatten()
    return similarities

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resumes = request.files.getlist('resumes')
        job_description = request.form['jd']
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

        match_scores = screen_resumes([r['text'] for r in results], job_description)

        for i, score in enumerate(match_scores):
            results[i]['match'] = round(float(score) * 100, 2)

        results.sort(key=lambda x: x['match'], reverse=True)
        df = pd.DataFrame(results)
        df = df[['file', 'email', 'phone', 'match']]
        df.to_csv('matched_resumes.csv', index=False)

        return render_template('results.html', tables=df.to_html(index=False))

    return render_template('index.html')

@app.route('/download')
def download():
    return send_file("matched_resumes.csv", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
