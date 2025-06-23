# Stock Anomaly Detection Services

This directory provides utilities for training and running anomaly detection models on stock data.

## Components

- `data_loader.py` – Fetch historical data from Yahoo Finance.
- `preprocess.py` – Create sequences with additional technical indicators (MA20, MA50, RSI).
- `model.py` – Original VQ-VAE model for reconstruction based anomaly detection.
- `wavenet_model.py` – WaveNet based model using dilated convolutions.
- `rl_trainer.py` – Lightweight reinforcement loop to update model weights after prediction.
- `evaluate.py` – Run models on new data and generate reports.
- `pattern_recognition.py` – Identify basic bullish/bearish patterns.
- `results/` – Sample output files.

A basic Flask app is provided in `../webapp/app.py` which exposes a `/predict` endpoint. It fetches the latest data every 5 minutes (only during market hours) and returns predictions while updating the model with new data.

Start the web service with:

```bash
cd webapp
gunicorn -k uvicorn.workers.UvicornWorker app:app
```
