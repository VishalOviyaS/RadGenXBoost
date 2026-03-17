import React from "react";
import { Bar, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(
  ArcElement,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend
);

function Dashboard({ result, downloadLink }) {

  // ✅ SAFETY CHECK (prevents crash)
  if (!result) {
    return <h2 style={{ color: "white", textAlign: "center" }}>Loading...</h2>;
  }

  const riskMap = ["Low", "Moderate", "High"];
  const fusion = riskMap[result.fusion_prediction];

  // ================= MODEL CONTRIBUTION =================
  const donutData = {
    labels: ["CT Model", "Genomic Model"],
    datasets: [
      {
        data: [
          result.ct_probabilities[0][1] * 100,
          result.genomic_probabilities[0][1] * 100
        ],
        backgroundColor: ["#22c55e", "#3b82f6"]
      }
    ]
  };

  // ================= FEATURE IMPORTANCE =================
  const featureLabels = result.feature_importance
    ? Object.keys(result.feature_importance)
    : ["Edge", "Texture", "Growth"];

  const featureValues = result.feature_importance
    ? Object.values(result.feature_importance)
    : [20, 50, 80];

  const featureData = {
    labels: featureLabels,
    datasets: [
      {
        label: "Tumor Metrics",
        data: featureValues,
        backgroundColor: "#f59e0b"
      }
    ]
  };

  return (
    <div className="dashboard-container">

      <h1 className="title">AI Tumor Analysis Dashboard</h1>

      {/* ================= IMAGES ================= */}
      <div className="image-section">

        <div className="image-card">
          <h3>Before Segmentation</h3>
          {result.ct_image ? (
            <img
              src={`data:image/png;base64,${result.ct_image}`}
              alt="CT"
            />
          ) : (
            <p>No image</p>
          )}
        </div>

        <div className="image-card">
          <h3>After Segmentation</h3>
          {result.segmented_image ? (
            <img
              src={`data:image/png;base64,${result.segmented_image}`}
              alt="Segmented"
            />
          ) : (
            <p>No segmented output</p>
          )}
        </div>

      </div>

      {/* ================= GRAPHS ================= */}
      <div className="graph-section">

        <div className="graph-card">
          <h3>Model Contribution</h3>
          <Doughnut data={donutData} />
        </div>

        <div className="graph-card">
          <h3>Tumor Metrics</h3>
          <Bar data={featureData} />
        </div>

      </div>

      {/* ================= RESULT ================= */}
      <div className="result-card">

        <h2>Tumor Aggressiveness: {fusion}</h2>

        <p>
          This prediction is based on radiomics and genomic features.
        </p>

        <p>
          <b>Clinical Insight:</b>{" "}
          {fusion === "High"
            ? "Highly aggressive tumor detected. Immediate treatment required."
            : fusion === "Moderate"
            ? "Moderate tumor growth. Regular monitoring needed."
            : "Low aggressiveness. Favorable prognosis."}
        </p>

      </div>

      {/* ================= DOWNLOAD ================= */}
      <div style={{ textAlign: "center", marginTop: "20px" }}>
        <a href={downloadLink} className="download-btn">
          Download Report
        </a>
      </div>

    </div>
  );
}

export default Dashboard;