const express = require('express');
const app = express();
const tf = require('@tensorflow/tfjs');
const cors = require('cors');
const fs = require('fs');

app.use(express.json());
app.use(cors());

let durationModel = null;
let entryModel = null;

async function loadModels() {
  try {
    durationModel = await tf.loadLayersModel('file://public/duration-model/model.json');
    entryModel = await tf.loadLayersModel('file://public/entry-model/model.json');
    console.log('✅ Models loaded successfully');
  } catch (err) {
    console.error('❌ Error loading models:', err.message);
  }
}
loadModels();

const fullScaler = JSON.parse(fs.readFileSync('mechineLearning/scaler/scaler.json', 'utf8'));
const shortScaler = JSON.parse(fs.readFileSync('mechineLearning/scaler/scalerdua.json', 'utf8'));

function standardScale(inputObj, scaler) {
  const fields = Object.keys(scaler);
  const values = fields.map(field => {
    if (!(field in inputObj)) throw new Error(`Missing field: ${field}`);
    if (typeof inputObj[field] !== 'number') throw new Error(`Invalid type for ${field}`);
    const { mean, scale } = scaler[field];
    if (scale === 0) throw new Error(`Scale for ${field} is zero`);
    return (inputObj[field] - mean) / scale;
  });
  return { values, fields };
}

// 🔴 Endpoint 2: Estimasi **Jam Masuk Poli**
app.post('/predict-entry', async (req, res) => {
  try {
    if (!entryModel) return res.status(503).json({ error: 'Entry model not loaded yet' });

    const inputObj = req.body.input;
    const { values } = standardScale(inputObj, shortScaler);

    const inputTensor = tf.tensor([values]);
    const prediction = entryModel.predict(inputTensor);
    const result = await prediction.data();

    const output = result[0];
    const targetMean = shortScaler["TriageToProviderStartTime"].mean;
    const targetStd = shortScaler["TriageToProviderStartTime"].scale;
    const predictedStartTime = (output * targetStd) + targetMean;

    res.json({
      prediction_scaled: output,
      prediction_minutes: predictedStartTime,
      prediction_seconds: predictedStartTime * 60
    });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// 🔵 Endpoint 1: Estimasi **Durasi Konsultasi Poli**
app.post('/predict-duration', async (req, res) => {
  try {
    if (!durationModel) return res.status(503).json({ error: 'Duration model not loaded yet' });

    const inputObj = req.body.input;
    const { values } = standardScale(inputObj, fullScaler);

    const inputTensor = tf.tensor([values]);
    const prediction = durationModel.predict(inputTensor);
    const result = await prediction.data();
    const scaledOutput = result[0];

    const targetMean = fullScaler["ConsultationDurationTime"].mean;
    const targetStd = fullScaler["ConsultationDurationTime"].scale;
    const durationInMinutes = (scaledOutput * targetStd) + targetMean;

    res.json({
      prediction_scaled: scaledOutput,
      prediction_minutes: durationInMinutes,
      prediction_seconds: durationInMinutes * 60
    });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
