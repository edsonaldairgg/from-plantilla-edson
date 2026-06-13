import express from 'express';
import {Storage} from '@google-cloud/storage';
import {parse} from 'csv-parse/sync';

const app = express();
const storage = new Storage();

let model = {};
let history = [];
let fileIndex = 0;
let blobs = null;

const learningRate = 0.0001;

const bucketName = 'bucket_131025';
const prefix = 'tlc_yellow_trips_2022/';
const limit = 1000;

function predict(x) {
  let y = model.bias || 0;
  for (const key of Object.keys(x)) {
    y += (model[key] || 0) * x[key];
  }
  return y;
}

function learn(x, y) {
  const pred = predict(x);
  const error = y - pred;

  model.bias = (model.bias || 0) + learningRate * error;

  for (const key of Object.keys(x)) {
    model[key] = (model[key] || 0) + learningRate * error * x[key];
  }

  return pred;
}

function extractFeatures(row) {
  const dist = Number(row.trip_distance || 0);
  const pass = Number(row.passenger_count || 0);

  return {
    dist,
    log_dist: Math.log1p(Math.max(dist, 0)),
    pass
  };
}

function calculateR2(yTrue, yPred) {
  const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  const ssRes = yTrue.reduce((sum, y, i) => sum + Math.pow(y - yPred[i], 2), 0);
  const ssTot = yTrue.reduce((sum, y) => sum + Math.pow(y - mean, 2), 0);
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

async function processFile(blobName) {
  const bucket = storage.bucket(bucketName);
  const file = bucket.file(blobName);

  const [content] = await file.download();
  const records = parse(content.toString(), {
    columns: true,
    skip_empty_lines: true
  });

  const yTrue = [];
  const yPred = [];

  let count = 0;

  for (const row of records) {
    if (count >= limit) break;

    const fare = Number(row.fare_amount);
    const dist = Number(row.trip_distance);
    const pass = Number(row.passenger_count);

    if (
      !Number.isFinite(fare) ||
      !Number.isFinite(dist) ||
      !Number.isFinite(pass)
    ) {
      continue;
    }

    if (
      fare < 2 || fare > 200 ||
      dist < 0.1 || dist > 50 ||
      pass < 1 || pass > 6
    ) {
      continue;
    }

    const x = extractFeatures(row);
    const pred = learn(x, fare);

    yTrue.push(fare);
    yPred.push(pred);

    count++;
  }

  return calculateR2(yTrue, yPred);
}

app.get('/', async (req, res) => {
  let html = `
    <h1>Aprendizaje en línea desde GCS</h1>
    <p>Bucket: <b>${bucketName}</b></p>
    <p>Prefijo: <b>${prefix}</b></p>
    <form action="/process" method="get">
      <button type="submit">Procesar siguiente archivo</button>
    </form>
    <hr>
    <h2>Estado actual</h2>
    <p>Archivos procesados: ${history.length}</p>
  `;

  if (history.length > 0) {
    html += `<p>Último R²: <b>${history[history.length - 1].toFixed(3)}</b></p>`;
    html += '<h3>Historial R²</h3><ul>';
    history.forEach((score, i) => {
      html += `<li>Archivo ${i + 1}: ${score.toFixed(3)}</li>`;
    });
    html += '</ul>';
  }

  res.send(html);
});

app.get('/process', async (req, res) => {
  try {
    if (blobs === null) {
      const [files] = await storage.bucket(bucketName).getFiles({prefix});
      blobs = files.filter(file => file.name.endsWith('.csv'));
    }

    if (fileIndex >= blobs.length) {
      return res.send(`
        <h1>Todos los archivos ya fueron procesados</h1>
        <a href="/">Volver</a>
      `);
    }

    const currentBlob = blobs[fileIndex];
    const score = await processFile(currentBlob.name);

    history.push(score);
    fileIndex++;

    res.send(`
      <h1>Archivo procesado</h1>
      <p>Archivo: <b>${currentBlob.name}</b></p>
      <p>R² acumulado: <b>${score.toFixed(3)}</b></p>
      <a href="/">Volver</a>
    `);

  } catch (error) {
    res.status(500).send(`
      <h1>Error</h1>
      <pre>${error.message}</pre>
      <a href="/">Volver</a>
    `);
  }
});

export default app;
