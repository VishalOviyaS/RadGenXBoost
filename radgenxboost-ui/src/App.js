import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import Dashboard from "./components/Dashboard";
import "./App.css";

function App() {

  const [step, setStep] = useState(0);
  const [result, setResult] = useState(null);
  const [downloadLink, setDownloadLink] = useState("");

  return (
    <div className="app-container">

      {/* STEP 0 - LANDING */}
      {step === 0 && (
        <div className="landing">
          <h1 className="title">RadGenXGBoost</h1>
          <p className="subtitle">
            AI-powered Lung Tumor Aggressiveness Prediction using
            Radiomics + Genomics Fusion
          </p>

          <button onClick={() => setStep(1)}>
            Get Started
          </button>
        </div>
      )}

      {/* STEP 1 - FORM */}
      {step === 1 && (
        <UploadForm
          onResult={(res, link) => {
            setResult(res);
            setDownloadLink(link);
            setStep(2);
          }}
        />
      )}

      {/* STEP 2 - DASHBOARD */}
      {step === 2 && (
        <Dashboard result={result} downloadLink={downloadLink} />
      )}

    </div>
  );
}

export default App;