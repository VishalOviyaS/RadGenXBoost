from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
import pydicom
import joblib
import io

from utils.radiomics import extract_radiomics_features
from utils.report_generator import generate_report

app = FastAPI()

# ---------------- LOAD MODELS ----------------
ct_model = joblib.load("models/ct_radiomics_model.pkl")
genomic_model = joblib.load("models/genomic_xgboost_model.pkl")
fusion_model = joblib.load("models/RadGenXGBoost_fusion_model.pkl")


# ---------------- HOME ROUTE ----------------
@app.get("/")
def home():
    return {"message": "RadGenXGBoost API is running"}


# ---------------- PREDICTION API ----------------
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

        print("CT feature shape:", ct_features.shape)

        ct_probs = ct_model.predict_proba(ct_features)

        print("CT probabilities:", ct_probs)


        # ================================
        # GENOMIC DATA PROCESSING
        # ================================
        contents = await genomic_file.read()

        genomic_df = pd.read_csv(io.BytesIO(contents))

        # keep only numeric columns
        genomic_df = genomic_df.select_dtypes(include=[np.number])

        print("Original genomic shape:", genomic_df.shape)

        # force 100 features (same as training)
        genomic_df = genomic_df.iloc[:, :100]

        print("Filtered genomic shape:", genomic_df.shape)

        genomic_data = genomic_df.values

        genomic_probs = genomic_model.predict_proba(genomic_data)

        print("Genomic probabilities:", genomic_probs)


        # ================================
        # FUSION MODEL
        # ================================
        fusion_input = np.hstack((ct_probs, genomic_probs))

        print("Fusion input shape:", fusion_input.shape)

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
        report_file = generate_report(result, ct_image=img)
        result["report_file"] = report_file


        report_path = generate_report(result, img)

        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename="RadGenXGBoost_Report.pdf"
        )


    except Exception as e:

        return {"error": str(e)}
    