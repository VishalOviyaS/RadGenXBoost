from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import pydicom
import joblib
import io
import os
import cv2
import base64

from utils.radiomics import extract_radiomics_features
from utils.report_generator import generate_report

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
ct_model = joblib.load(os.path.join(BASE_DIR, "models/ct_radiomics_model.pkl"))
genomic_model = joblib.load(os.path.join(BASE_DIR, "models/genomic_xgboost_model.pkl"))
fusion_model = joblib.load(os.path.join(BASE_DIR, "models/RadGenXGBoost_fusion_model.pkl"))

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "RadGenXGBoost API is running"}

# ---------------- DOWNLOAD ----------------
@app.get("/download/{file_name}")
def download_report(file_name: str):
    file_path = os.path.join(REPORT_DIR, file_name)
    return FileResponse(file_path, media_type="application/pdf", filename="RadGenXGBoost_Report.pdf")

# ---------------- PREDICT ----------------
@app.post("/predict")
async def predict(ct_file: UploadFile = File(...), genomic_file: UploadFile = File(...)):

    try:
        # ================================
        # LOAD CT IMAGE
        # ================================
        dicom = pydicom.dcmread(ct_file.file)
        img = dicom.pixel_array

        # Normalize image
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
        img_norm = (img_norm * 255).astype(np.uint8)

        # ================================
        # RADIOMICS FEATURES
        # ================================
        ct_features = extract_radiomics_features(img)
        ct_features_array = np.array(ct_features).reshape(1, -1)

        ct_probs = ct_model.predict_proba(ct_features_array)

        # ================================
        # GENOMIC DATA
        # ================================
        contents = await genomic_file.read()
        genomic_df = pd.read_csv(io.BytesIO(contents))

        genomic_df = genomic_df.select_dtypes(include=[np.number])
        genomic_df = genomic_df.iloc[:, :100]

        genomic_data = genomic_df.iloc[0].values.reshape(1, -1)
        genomic_probs = genomic_model.predict_proba(genomic_data)

        # ================================
        # FUSION MODEL
        # ================================
        fusion_input = np.hstack((ct_probs, genomic_probs))
        fusion_pred = fusion_model.predict(fusion_input)

        # ================================
        # SEGMENTATION (IMPROVED)
        # ================================
        blur = cv2.GaussianBlur(img_norm, (5,5), 0)

        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 2)

        # ================================
        # CONVERT IMAGES TO BASE64
        # ================================
        _, buffer1 = cv2.imencode(".png", img_norm)
        ct_base64 = base64.b64encode(buffer1).decode("utf-8")

        _, buffer2 = cv2.imencode(".png", overlay)
        seg_base64 = base64.b64encode(buffer2).decode("utf-8")

        # ================================
        # FEATURE METRICS (REAL + NORMALIZED)
        # ================================
        feature_names = [
            "Mean Intensity",
            "Std Deviation",
            "Skewness",
            "Kurtosis",
            "Contrast",
            "Energy",
            "Homogeneity",
            "Correlation",
            "Dissimilarity",
            "Variance"
        ]

        feature_values = np.array(ct_features)

        # Normalize to 0–100 (for visualization)
        feature_values = 100 * (feature_values - feature_values.min()) / (
            feature_values.max() - feature_values.min() + 1e-6
        )

        feature_importance = dict(zip(feature_names, feature_values.tolist()))

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
        # GENERATE REPORT
        # ================================
        report_filename = "prediction_report.pdf"
        report_path = os.path.join(REPORT_DIR, report_filename)
        generate_report(result, img, report_path)

        # ================================
        # FINAL RESPONSE
        # ================================
        return {
            "message": "Prediction completed",
            "download_url": f"http://127.0.0.1:8000/download/{report_filename}",
            "result": result,
            "ct_image": ct_base64,
            "segmented_image": seg_base64,
            "feature_importance": feature_importance
        }

    except Exception as e:
        return {"error": str(e)}