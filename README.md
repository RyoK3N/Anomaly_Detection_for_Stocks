# Stock Anomaly Detection

This project provides utilities for detecting anomalies in stock price data using a Vector Quantized VAE and a WaveNet predictor. It also exposes a small web service for real-time inference.

## Architecture

```
           +-------------+
           | data_loader |
           +------+------+
                  |
                  v
           +-------------+
           | preprocess  |-- technical_indicators
           +------+------+
                  |
                  v
+--------+   +-----------+
| model  |<--| wavenet   |
+---+----+   +-----------+
    |             |
    v             v
 evaluate      rl_trainer
    |
    v
 pattern_recognition
```

- **data_loader** fetches historical data from Yahoo Finance.
- **preprocess** scales data and adds indicators via `technical_indicators`.
- **model** implements a VQ‑VAE for anomaly detection.
- **wavenet_model** provides a WaveNet-based predictor.
- **rl_trainer** performs a lightweight reinforcement-style weight update.
- **evaluate** runs the trained model on new data and uses `pattern_recognition` to label results.
- **webapp** exposes the prediction endpoint via Flask/Gunicorn.

## Quick start

```bash
make lint
make test
make run-ui
```

See `services/README.md` for more details on individual modules.
