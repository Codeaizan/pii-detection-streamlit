import streamlit as st
import pdfplumber
import joblib
import json
import tempfile
import os
import re

# -------------------------------------------------
# Load Model Artifacts
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("pii_svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_mapping = joblib.load("label_mapping.pkl")
    return model, vectorizer, label_mapping

model, vectorizer, label_mapping = load_artifacts()

# -------------------------------------------------
# Regex-based PII Detection Gate (CRITICAL FIX)
# -------------------------------------------------
AADHAAR_REGEX = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
PASSPORT_REGEX = re.compile(r"\b[A-Z][0-9]{7}\b")

def is_potential_pii(line):
    return (
        AADHAAR_REGEX.search(line)
        or PAN_REGEX.search(line)
        or PASSPORT_REGEX.search(line)
    )

# -------------------------------------------------
# PDF Text Extraction
# -------------------------------------------------
def extract_text_from_pdf(pdf_path):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split("\n"))
    return [line.strip() for line in lines if line.strip()]

# -------------------------------------------------
# PII Detection + Classification (FIXED)
# -------------------------------------------------
def detect_pii(lines):
    # Step 1: Regex gate
    candidate_lines = [line for line in lines if is_potential_pii(line)]

    pii_output = {
        "AADHAAR": [],
        "PAN": [],
        "PASSPORT": []
    }

    # If no candidates, return empty result
    if not candidate_lines:
        return pii_output

    # Step 2: ML classification only on candidates
    X = vectorizer.transform(candidate_lines)
    predictions = model.predict(X)

    for line, label in zip(candidate_lines, predictions):
        if label in pii_output:
            pii_output[label].append(line)

    return pii_output

# -------------------------------------------------
# JSON Output Builder (REWRITTEN)
# -------------------------------------------------
def build_json(document_name, pii_data):
    return {
        "document_name": document_name,
        "summary": {
            "aadhaar_count": len(pii_data["AADHAAR"]),
            "pan_count": len(pii_data["PAN"]),
            "passport_count": len(pii_data["PASSPORT"]),
            "total_pii_items": (
                len(pii_data["AADHAAR"]) +
                len(pii_data["PAN"]) +
                len(pii_data["PASSPORT"])
            )
        },
        "detected_pii": {
            label_mapping["AADHAAR"]: pii_data["AADHAAR"],
            label_mapping["PAN"]: pii_data["PAN"],
            label_mapping["PASSPORT"]: pii_data["PASSPORT"]
        }
    }

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="PII Detection System",
    layout="centered"
)

st.title("AI-Based PII Detection in Documents")
st.subheader("Upload a PDF to detect government-issued PII")

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded successfully.")

    if st.button("Detect PII"):
        with st.spinner("Analyzing document..."):
            lines = extract_text_from_pdf(pdf_path)
            pii_data = detect_pii(lines)
            json_output = build_json(uploaded_file.name, pii_data)

        st.subheader("Detected PII (JSON Output)")
        st.json(json_output)

        st.download_button(
            label="Download JSON",
            data=json.dumps(json_output, indent=4),
            file_name="detected_pii.json",
            mime="application/json"
        )

    os.remove(pdf_path)
