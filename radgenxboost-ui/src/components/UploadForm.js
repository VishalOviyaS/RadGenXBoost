import React, { useState } from "react";

function UploadForm({ onResult }) {

  const [patientId, setPatientId] = useState("");
  const [ctFile, setCtFile] = useState(null);
  const [genomicFile, setGenomicFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {

    e.preventDefault();

    if (!patientId || !ctFile || !genomicFile) {
      alert("Fill all fields");
      return;
    }

    const formData = new FormData();

    formData.append("ct_file", ctFile);
    formData.append("genomic_file", genomicFile);

    setLoading(true);

    try {

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      onResult(data.result, data.download_url);

    } catch (err) {
      alert("Error");
    }

    setLoading(false);
  };

  return (
    <div className="card">

      <h2>Patient Analysis</h2>

      <form onSubmit={handleSubmit}>

        <label>Patient ID</label>
        <input
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
        />

        <label>Upload CT (.dcm)</label>
        <input type="file" accept=".dcm"
          onChange={(e) => setCtFile(e.target.files[0])}
        />

        <label>Upload Genomic (.csv)</label>
        <input type="file" accept=".csv"
          onChange={(e) => setGenomicFile(e.target.files[0])}
        />

        <button>
          {loading ? "Analyzing..." : "Predict Analysis"}
        </button>

      </form>

    </div>
  );
}

export default UploadForm;