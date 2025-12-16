import streamlit as st
import pdfplumber
import joblib
import json
import tempfile
import os

# ----------------------------
# Load Model Artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("pii_svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_mapping = joblib.load("label_mapping.pkl")
    return model, vectorizer, label_mapping

model, vectorizer, label_mapping = load_artifacts()

# ----------------------------
# PDF Text Extraction
# ----------------------------
def extract_text_from_pdf(pdf_path):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split("\n"))
    return [line.strip() for line in lines if line.strip()]

# ----------------------------
# PII Detection
# ----------------------------
def detect_pii(lines):
    X = vectorizer.transform(lines)
    predictions = model.predict(X)

    pii_output = {
        "AADHAAR": [],
        "PAN": [],
        "PASSPORT": []
    }

    for line, label in zip(lines, predictions):
        if label in pii_output:
            pii_output[label].append(line)

    return pii_output

# ----------------------------
# JSON Builder
# ----------------------------
def build_json(document_name, pii_data):
    return {
        "document_name": document_name,
        "detected_pii": {
            label_mapping[k]: v for k, v in pii_data.items()
        }
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="PII Detection System",
    layout="centered"
)

st.title("AI-Based PII Detection in Documents")
st.subheader("Upload a PDF to detect government-issued PII")

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully.")

    if st.button("Detect PII"):
        with st.spinner("Analyzing document..."):
            lines = extract_text_from_pdf(pdf_path)
            pii_data = detect_pii(lines)
            json_output = build_json(uploaded_file.name, pii_data)

        st.subheader("Detected PII (JSON Output)")
        st.json(json_output)

        json_str = json.dumps(json_output, indent=4)

        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="detected_pii.json",
            mime="application/json"
        )

    os.remove(pdf_path)
