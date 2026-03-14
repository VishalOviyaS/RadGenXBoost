from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import pydicom
import joblib
import io
import os

from utils.radiomics import extract_radiomics_features
from utils.report_generator import generate_report

app = FastAPI()

# ---------------- CORS (for frontend connection) ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ct_model = joblib.load(os.path.join(BASE_DIR, "models/ct_radiomics_model.pkl"))
genomic_model = joblib.load(os.path.join(BASE_DIR, "models/genomic_xgboost_model.pkl"))
fusion_model = joblib.load(os.path.join(BASE_DIR, "models/RadGenXGBoost_fusion_model.pkl"))


# ---------------- HOME ROUTE ----------------
@app.get("/")
def home():
    return {"message": "RadGenXGBoost API is running"}


# ---------------- PREDICTION ROUTE ----------------
@app.post("/predict")
async def predict(ct_file: UploadFile = File(...), genomic_file: UploadFile = File(...)):

    try:

        # ================================
        # CT IMAGE PROCESSING
        # ================================
        dicom = pydicom.dcmread(ct_file.file)
        img = dicom.pixel_array

        ct_features = extract_radiomics_features(img)
        ct_features = np.array(ct_features).reshape(1, -1)

        ct_probs = ct_model.predict_proba(ct_features)

        # ================================
        # GENOMIC DATA PROCESSING
        # ================================
        contents = await genomic_file.read()
        genomic_df = pd.read_csv(io.BytesIO(contents))

        # Keep only numeric columns
        genomic_df = genomic_df.select_dtypes(include=[np.number])

        # Limit to first 100 features
        genomic_df = genomic_df.iloc[:, :100]

        # Ensure single row input
        genomic_data = genomic_df.iloc[0].values.reshape(1, -1)

        genomic_probs = genomic_model.predict_proba(genomic_data)

        # ================================
        # FUSION MODEL
        # ================================
        fusion_input = np.hstack((ct_probs, genomic_probs))
        fusion_pred = fusion_model.predict(fusion_input)

        # ================================
        # RESULT
        # ================================
        result = {
            "ct_prediction": int(np.argmax(ct_probs)),
            "genomic_prediction": int(np.argmax(genomic_probs)),
            "fusion_prediction": int(fusion_pred[0]),
            "ct_probabilities": ct_probs.tolist(),
            "genomic_probabilities": genomic_probs.tolist()
        }

        # ================================
        # GENERATE PDF REPORT
        # ================================
        report_path = generate_report(result, img)

        return {
            "message": "Prediction completed",
            "download_url": f"https://radgenxboost.onrender.com/download/{report_path}"
        }

    except Exception as e:
        return {"error": str(e)}