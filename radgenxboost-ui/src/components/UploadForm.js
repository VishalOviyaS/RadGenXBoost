import React, { useState } from "react";

function UploadForm() {

  const [ctFile, setCtFile] = useState(null);
  const [genomicFile, setGenomicFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [downloadLink, setDownloadLink] = useState("");

  const handleSubmit = async (e) => {

    e.preventDefault();

    if (!ctFile || !genomicFile) {
      alert("Upload both CT and Genomic files");
      return;
    }

    const formData = new FormData();

    formData.append("ct_file", ctFile);
    formData.append("genomic_file", genomicFile);

    setLoading(true);

    try {

      const response = await fetch(
        "https://radgenxboost.onrender.com/predict",
        {
          method: "POST",
          body: formData
        }
      );

      const data = await response.json();

      setDownloadLink(data.download_url);

    } catch (error) {
      alert("Prediction failed");
    }

    setLoading(false);
  };

  return (
    <div className="card">

      <form onSubmit={handleSubmit}>

        <label>Upload CT Scan (.dcm)</label>

        <input
          type="file"
          accept=".dcm"
          onChange={(e) => setCtFile(e.target.files[0])}
        />

        <label>Upload Genomic CSV</label>

        <input
          type="file"
          accept=".csv"
          onChange={(e) => setGenomicFile(e.target.files[0])}
        />

        <button type="submit">

          {loading ? "Processing..." : "Predict"}

        </button>

      </form>

      {downloadLink && (
        <a href={downloadLink} target="_blank" rel="noreferrer">
          Download AI Report
        </a>
      )}

    </div>
  );
}

export default UploadForm;