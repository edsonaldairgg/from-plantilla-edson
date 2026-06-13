import express from 'express';
import {Storage} from '@google-cloud/storage';
import {parse} from 'csv-parse/sync';

const app = express();

const projectId = 'grandes-datos-edson-guerrero';
const bucketName = 'bucket-grandesdatos-eagg';
const prefix = 'tlc_yellow_trips_2022/';
const modelPath = 'models/model_incremental_node.json';
const limit = 1000;
const learningRate = 0.0001;

const storage = new Storage({projectId});

let model = {};
let history = [];
let fileIndex = 0;
let blobs = null;

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

  let hour = 0;
  let dow = 0;
  let isWeekend = 0;

  const dateValue =
    row.tpep_pickup_datetime ||
    row.lpep_pickup_datetime ||
    row.pickup_datetime;

  if (dateValue) {
    const date = new Date(dateValue);
    if (!Number.isNaN(date.getTime())) {
      hour = date.getHours();
      dow = date.getDay();
      isWeekend = dow === 0 || dow === 6 ? 1 : 0;
    }
  }

  return {
    dist,
    log_dist: Math.log1p(Math.max(dist, 0)),
    pass,
    hour,
    dow,
    is_weekend: isWeekend
  };
}

function calculateR2(yTrue, yPred) {
  if (yTrue.length === 0) return 0;

  const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;

  const ssRes = yTrue.reduce((sum, y, i) => {
    return sum + Math.pow(y - yPred[i], 2);
  }, 0);

  const ssTot = yTrue.reduce((sum, y) => {
    return sum + Math.pow(y - mean, 2);
  }, 0);

  if (ssTot === 0) return 0;

  return 1 - ssRes / ssTot;
}

async function saveModelToGCS() {
  const payload = JSON.stringify({
    model,
    history,
    fileIndex,
    updatedAt: new Date().toISOString()
  });

  await storage
    .bucket(bucketName)
    .file(modelPath)
    .save(payload, {
      contentType: 'application/json'
    });
}

async function loadModelFromGCS() {
  try {
    const file = storage.bucket(bucketName).file(modelPath);
    const [exists] = await file.exists();

    if (!exists) return;

    const [content] = await file.download();
    const saved = JSON.parse(content.toString());

    model = saved.model || {};
    history = saved.history || [];
    fileIndex = saved.fileIndex || 0;
  } catch (error) {
    console.error('No se pudo cargar el modelo:', error.message);
  }
}

async function listCsvFiles() {
  const [files] = await storage.bucket(bucketName).getFiles({prefix});
  return files
    .filter(file => file.name.toLowerCase().endsWith('.csv'))
    .sort((a, b) => a.name.localeCompare(b.name));
}

async function processFile(blobName) {
  const file = storage.bucket(bucketName).file(blobName);
  const [content] = await file.download();

  const records = parse(content.toString(), {
    columns: true,
    skip_empty_lines: true,
    relax_column_count: true
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
      fare < 2 ||
      fare > 200 ||
      dist < 0.1 ||
      dist > 50 ||
      pass < 1 ||
      pass > 6
    ) {
      continue;
    }

    const x = extractFeatures(row);
    const pred = learn(x, fare);

    yTrue.push(fare);
    yPred.push(pred);

    count++;
  }

  const r2 = calculateR2(yTrue, yPred);

  return {
    r2,
    rowsProcessed: count
  };
}

function renderPage(message = '') {
  const lastR2 = history.length > 0 ? history[history.length - 1] : null;

  let historyHtml = '<p>Aún no se ha procesado ningún archivo.</p>';

  if (history.length > 0) {
    historyHtml = '<ol>';
    history.forEach((score, i) => {
      historyHtml += `<li>Archivo ${i + 1}: R² = ${score.toFixed(3)}</li>`;
    });
    historyHtml += '</ol>';
  }

  return `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Aprendizaje en línea</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background: #f6f8fa;
            color: #202124;
          }
          .card {
            background: white;
            padding: 28px;
            border-radius: 14px;
            max-width: 900px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
          }
          button {
            background: #1a73e8;
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
          }
          button.secondary {
            background: #d93025;
          }
          .message {
            background: #e8f0fe;
            padding: 14px;
            border-radius: 8px;
            margin: 18px 0;
          }
          code {
            background: #f1f3f4;
            padding: 2px 6px;
            border-radius: 4px;
          }
          li {
            margin-bottom: 6px;
          }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Aprendizaje en línea desde GCS</h1>
          <p>Procesamiento incremental archivo por archivo desde Google Cloud Storage.</p>

          <p><b>Proyecto:</b> <code>${projectId}</code></p>
          <p><b>Bucket:</b> <code>${bucketName}</code></p>
          <p><b>Prefijo:</b> <code>${prefix}</code></p>
          <p><b>Filas por archivo:</b> ${limit}</p>

          ${message ? `<div class="message">${message}</div>` : ''}

          <form action="/process" method="get">
            <button type="submit">Procesar siguiente archivo</button>
          </form>

          <br>

          <form action="/reset" method="post">
            <button class="secondary" type="submit">Reiniciar entrenamiento</button>
          </form>

          <hr>

          <h2>Estado actual del modelo</h2>
          <p><b>Archivos procesados:</b> ${history.length}</p>
          <p><b>Siguiente archivo:</b> ${fileIndex + 1}</p>
          <p><b>R² actual:</b> ${lastR2 === null ? '0.000' : lastR2.toFixed(3)}</p>

          <h3>Evolución del R²</h3>
          ${historyHtml}

          <p><small>Modelo guardado en GCS como <code>${modelPath}</code></small></p>
        </div>
      </body>
    </html>
  `;
}

app.use(express.urlencoded({extended: true}));

app.get('/', async (req, res) => {
  await loadModelFromGCS();
  res.send(renderPage());
});

app.get('/process', async (req, res) => {
  try {
    await loadModelFromGCS();

    if (blobs === null) {
      blobs = await listCsvFiles();
    }

    if (blobs.length === 0) {
      return res.send(renderPage(
        'No se encontraron archivos CSV. Revisa el bucket o el prefijo.'
      ));
    }

    if (fileIndex >= blobs.length) {
      return res.send(renderPage(
        'Todos los archivos ya fueron procesados.'
      ));
    }

    const currentBlob = blobs[fileIndex];
    const result = await processFile(currentBlob.name);

    history.push(result.r2);
    fileIndex++;

    await saveModelToGCS();

    res.send(renderPage(
      `Archivo procesado: <code>${currentBlob.name}</code><br>
       Filas válidas procesadas: <b>${result.rowsProcessed}</b><br>
       R² acumulado: <b>${result.r2.toFixed(3)}</b>`
    ));
  } catch (error) {
    res.status(500).send(renderPage(
      `Error al procesar archivo:<br><code>${error.message}</code>`
    ));
  }
});

app.post('/reset', async (req, res) => {
  try {
    model = {};
    history = [];
    fileIndex = 0;
    blobs = null;

    const file = storage.bucket(bucketName).file(modelPath);
    const [exists] = await file.exists();

    if (exists) {
      await file.delete();
    }

    res.send(renderPage('Entrenamiento reiniciado correctamente.'));
  } catch (error) {
    res.status(500).send(renderPage(
      `Error al reiniciar:<br><code>${error.message}</code>`
    ));
  }
});

export default app;
