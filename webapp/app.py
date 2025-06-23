import asyncio
from datetime import datetime, time as dtime

from flask import Flask, jsonify
import pandas as pd
import torch
import yfinance as yf

from services.preprocess import preprocess_data
from services.wavenet_model import WaveNet, load_wavenet
from services.rl_trainer import rl_update

app = Flask(__name__)

SYMBOL = "AAPL"
INTERVAL = "5m"
SEQ_LENGTH = 60
PRED_STEPS = 10
model = load_wavenet(input_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def market_open(now):
    if now.weekday() >= 5:
        return False
    open_time = dtime(9, 30)
    close_time = dtime(16, 0)
    return open_time <= now.time() <= close_time


async def fetch_data():
    end = datetime.utcnow()
    start = end - pd.Timedelta(minutes=SEQ_LENGTH + PRED_STEPS * 5)
    df = yf.download(SYMBOL, start=start, end=end, interval=INTERVAL, progress=False)
    return df


async def predict_and_update():
    if not market_open(datetime.utcnow()):
        return {"message": "Market closed"}
    df = await fetch_data()
    if df.empty:
        return {"message": "No data"}
    sequences, scaler = preprocess_data(df, SEQ_LENGTH)
    tensor = torch.tensor(sequences, dtype=torch.float32).permute(0, 2, 1).to(model.device)
    with torch.no_grad():
        pred = model(tensor)
    # RL update will run for 10 steps asynchronously
    await rl_update(model, optimizer, pred, tensor)
    return {"prediction": pred[-1, -1, -PRED_STEPS:].cpu().tolist()}


@app.route("/predict")
async def predict():
    result = await predict_and_update()
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
