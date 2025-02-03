from flask import Flask, request, render_template
import pickle
from PyPDF2 import PdfReader
import re
import pandas as pd


app=Flask(__name__)
# Load models===========================================================================================================
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))


# Clean resume==========================================================================================================
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Prediction and Category Name
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

# Prediction and Category Name
def job_recommendation(resume_text):
    resume_text= cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job


def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text
  
def extract_contact_number_from_resume(text):
    contact_number = None

    # Use regex pattern to find a potential contact number
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()

    return contact_number
  
def extract_email_from_resume(text):
    email = None

    # Use regex pattern to find a potential email address
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email


def extract_skills_from_resume(text, skills_list):
    skills_set = set(skills_list)
    skills_found = set()
    for skill in skills_set:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            skills_found.add(skill)
    if skills_found:
        return list(set(skill.lower() for skill in skills_found))
    else:
        return None


def load_education_keywords(csv_file):
    """Charge les mots-clés des formations depuis un fichier CSV formaté en une seule ligne."""
    with open(csv_file, "r", encoding="utf-8") as file:
        line = file.readline().strip()  # Lire la seule ligne du fichier et enlever les espaces inutiles
    
    # Supprimer les apostrophes et diviser par la virgule
    education_keywords = [keyword.strip().strip("'") for keyword in line.split(",")]
    
    return list(set(education_keywords))  # Supprime les doublons

def extract_education_from_resume(text, csv_file):
    """Extrait les formations du texte en fonction des mots-clés chargés depuis le CSV"""
    education = []
    
    # Charger la liste des formations depuis le fichier CSV
    education_keywords = load_education_keywords(csv_file)
    
    for keyword in education_keywords:
        # Si le mot-clé contient un espace (comme "Data Science"), ne pas utiliser \b
        if " " in keyword:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)  # Recherche insensible à la casse
        else:
            pattern = re.compile(r"\b{}\b".format(re.escape(keyword)), re.IGNORECASE)  # Ajout de \b pour les mots simples
        
        if pattern.search(text):
            education.append(keyword.lower())  # Ajoute la formation en minuscule
    
    return list(set(education))  # Supprime les doublons

# Exemple d'utilisation
csv_file = "./data/Education.csv"

def extract_name_from_resume(text):
    name = None

    # Use regex pattern to find a potential name
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()

    return name

@app.route('/')


def resume():
  return render_template('resume.html')

@app.route('/pred', methods=['POST'])
def pred():
    # Process the PDF or TXT file and make prediction
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename
        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")
        skills_df = pd.read_csv('./data/Skills.csv')
        skills_list = skills_df['skill'].tolist()
        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)

        extracted_skills = extract_skills_from_resume(text,skills_list)
        extracted_education = extract_education_from_resume(text,csv_file)
        name = extract_name_from_resume(text)

        return render_template('resume.html', predicted_category=predicted_category,recommended_job=recommended_job,
                               phone=phone,name=name,email=email,extracted_skills=extracted_skills,extracted_education=extracted_education)
    else:
        return render_template("resume.html", message="No resume file uploaded.")
     
  

if __name__ == '__main__':
  app.run(debug=True)
