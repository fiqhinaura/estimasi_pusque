const express = require('express');
const app = express();
const tf = require('@tensorflow/tfjs-node'); // ⬅️ Ganti dari @tensorflow/tfjs ke tfjs-node
const cors = require('cors');
const fs = require('fs');
const path = require('path');

// Middleware
app.use(express.json());
app.use(cors());
app.use('/public', express.static(path.join(__dirname, 'public'))); // ⬅️ serve model dari folder public
app.use(express.static(path.join(__dirname, 'mechineLearning')));

let durationModel = null;
let entryModel = null;

const loadModels = async () => {
  try {
    const baseURL = process.env.BASE_URL || `http://localhost:${PORT}`;
    console.log(`🔁 Loading models from ${baseURL}`);

    durationModel = await tf.loadLayersModel(`${baseURL}/public/duration-model/model.json`);
    entryModel = await tf.loadLayersModel(`${baseURL}/public/entry-model/model.json`);

    console.log('✅ Models loaded successfully');
  } catch (err) {
    console.error('❌ Error loading models:', err);
  }
};

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

// 🔵 Endpoint: Prediksi durasi konsultasi
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

// 🔴 Endpoint: Prediksi jam masuk poli
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

// Halaman utama
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'mechineLearning', 'index.html'));
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', async () => {
  console.log(`🚀 Server running on http://0.0.0.0:${PORT}`);
  await loadModels(); // ⬅️ panggil loadModels saat server nyala
});
