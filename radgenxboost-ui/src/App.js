import React from "react";
import UploadForm from "./components/UploadForm";
import "./App.css";

function App() {
  return (
    <div className="app-container">

      <h1 className="title">
        RadGenXGBoost
      </h1>

      <p className="subtitle">
        AI Lung Tumor Aggressiveness Prediction
      </p>

      <UploadForm />

    </div>
  );
}

export default App;