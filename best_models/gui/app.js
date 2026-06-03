const labels = ["Mild", "Normal", "Severe"];
const bestCnnWeight = 0.5;

let rows = [];
let grid = [];
let currentRows = [];

const fmt = (value, digits = 4) => Number(value).toFixed(digits);

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
  });
}

function predictionFor(row, cnnWeight) {
  const rfWeight = 1 - cnnWeight;
  const probs = labels.map((label) => {
    return cnnWeight * Number(row[`cnn_prob_${label}`]) + rfWeight * Number(row[`rf_prob_${label}`]);
  });
  const maxIndex = probs.indexOf(Math.max(...probs));
  return { probs, pred: labels[maxIndex] };
}

function classMetrics(yTrue, yPred, label) {
  let tp = 0, fp = 0, fn = 0;
  yTrue.forEach((truth, index) => {
    const pred = yPred[index];
    if (truth === label && pred === label) tp += 1;
    if (truth !== label && pred === label) fp += 1;
    if (truth === label && pred !== label) fn += 1;
  });
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);
  return { precision, recall, f1 };
}

function balancedAccuracy(yTrue, yPred) {
  const recalls = labels.map((label) => classMetrics(yTrue, yPred, label).recall);
  return recalls.reduce((sum, value) => sum + value, 0) / recalls.length;
}

function macroF1(yTrue, yPred) {
  const f1s = labels.map((label) => classMetrics(yTrue, yPred, label).f1);
  return f1s.reduce((sum, value) => sum + value, 0) / f1s.length;
}

function recompute() {
  const cnnWeight = Number(document.getElementById("cnnWeight").value);
  const rfWeight = 1 - cnnWeight;
  document.getElementById("cnnWeightText").textContent = fmt(cnnWeight, 2);
  document.getElementById("rfWeight").value = fmt(rfWeight, 2);

  currentRows = rows.map((row) => {
    const blend = predictionFor(row, cnnWeight);
    return { ...row, blendProbs: blend.probs, blend_pred_live: blend.pred };
  });

  const yTrue = currentRows.map((row) => row.y_true);
  const yPred = currentRows.map((row) => row.blend_pred_live);
  const accuracy = yTrue.filter((truth, index) => truth === yPred[index]).length / yTrue.length;

  document.getElementById("nRows").textContent = String(currentRows.length);
  document.getElementById("accuracy").textContent = fmt(accuracy);
  document.getElementById("balancedAccuracy").textContent = fmt(balancedAccuracy(yTrue, yPred));
  document.getElementById("macroF1").textContent = fmt(macroF1(yTrue, yPred));

  renderTable();
  drawChart(cnnWeight);
}

function renderTable() {
  const caseFilter = document.getElementById("caseFilter").value.trim();
  const labelFilter = document.getElementById("labelFilter").value;
  const filtered = currentRows.filter((row) => {
    const caseOk = !caseFilter || String(row["Case ID"]).includes(caseFilter);
    const labelOk = !labelFilter || row.blend_pred_live === labelFilter;
    return caseOk && labelOk;
  });

  document.getElementById("shownRows").textContent = `${filtered.length} lignes affichees`;
  const body = document.getElementById("predBody");
  body.innerHTML = filtered.slice(0, 250).map((row) => {
    const cells = [
      row["Case ID"],
      `<span class="pill ${row.y_true}">${row.y_true}</span>`,
      `<span class="pill ${row.blend_pred_live}">${row.blend_pred_live}</span>`,
      ...row.blendProbs.map((value) => fmt(value, 3)),
      ...labels.map((label) => fmt(row[`cnn_prob_${label}`], 3)),
      ...labels.map((label) => fmt(row[`rf_prob_${label}`], 3)),
    ];
    return `<tr>${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr>`;
  }).join("");
}

function drawChart(currentWeight) {
  const canvas = document.getElementById("weightChart");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 58, right: 24, top: 22, bottom: 48 };
  const values = grid.flatMap((row) => [Number(row.accuracy), Number(row.macro_f1)]);
  const minY = Math.max(0, Math.min(...values) - 0.04);
  const maxY = Math.min(1, Math.max(...values) + 0.04);
  const x = (w) => pad.left + Number(w) * (width - pad.left - pad.right);
  const y = (score) => height - pad.bottom - ((score - minY) / (maxY - minY)) * (height - pad.top - pad.bottom);

  ctx.strokeStyle = "#d9e0e8";
  ctx.lineWidth = 1;
  ctx.font = "13px Segoe UI, Arial";
  ctx.fillStyle = "#657284";
  for (let i = 0; i <= 4; i += 1) {
    const score = minY + (maxY - minY) * i / 4;
    const yy = y(score);
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(width - pad.right, yy);
    ctx.stroke();
    ctx.fillText(fmt(score, 2), 10, yy + 4);
  }

  function lineFor(key, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    grid.forEach((row, index) => {
      const xx = x(row.cnn_weight);
      const yy = y(row[key]);
      if (index === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
    grid.forEach((row) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x(row.cnn_weight), y(row[key]), 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  lineFor("accuracy", "#2166ac");
  lineFor("macro_f1", "#b2182b");

  ctx.strokeStyle = "#1b7837";
  ctx.setLineDash([8, 6]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x(currentWeight), pad.top);
  ctx.lineTo(x(currentWeight), height - pad.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#18212f";
  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    ctx.fillText(fmt(tick, 2), x(tick) - 14, height - 18);
  });

  ctx.fillStyle = "#2166ac";
  ctx.fillText("Accuracy", width - 190, 30);
  ctx.fillStyle = "#b2182b";
  ctx.fillText("Macro-F1", width - 100, 30);
}

function exportCsv() {
  const headers = [
    "Case ID", "y_true", "blend_pred",
    "blend_prob_Mild", "blend_prob_Normal", "blend_prob_Severe",
    "cnn_prob_Mild", "cnn_prob_Normal", "cnn_prob_Severe",
    "rf_prob_Mild", "rf_prob_Normal", "rf_prob_Severe",
  ];
  const lines = [headers.join(",")];
  currentRows.forEach((row) => {
    lines.push([
      row["Case ID"],
      row.y_true,
      row.blend_pred_live,
      ...row.blendProbs.map((value) => fmt(value, 8)),
      ...labels.map((label) => fmt(row[`cnn_prob_${label}`], 8)),
      ...labels.map((label) => fmt(row[`rf_prob_${label}`], 8)),
    ].join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "tmj_blender_predictions.csv";
  a.click();
  URL.revokeObjectURL(url);
}

async function init() {
  const [predText, gridText] = await Promise.all([
    fetch("../blender_50_50/final_50_50_oof_predictions.csv").then((response) => response.text()),
    fetch("../blender_50_50/simple_weight_blender_grid.csv").then((response) => response.text()),
  ]);
  rows = parseCsv(predText);
  grid = parseCsv(gridText).sort((a, b) => Number(a.cnn_weight) - Number(b.cnn_weight));
  document.getElementById("status").textContent = "Prêt";
  recompute();
}

document.getElementById("cnnWeight").addEventListener("input", recompute);
document.getElementById("caseFilter").addEventListener("input", renderTable);
document.getElementById("labelFilter").addEventListener("change", renderTable);
document.getElementById("resetBest").addEventListener("click", () => {
  document.getElementById("cnnWeight").value = String(bestCnnWeight);
  recompute();
});
document.getElementById("exportCsv").addEventListener("click", exportCsv);

init().catch((error) => {
  document.getElementById("status").textContent = "Erreur";
  console.error(error);
});
