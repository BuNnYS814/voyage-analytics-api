# Voyage Analytics API

Flight price prediction API powered by XGBoost and FastAPI.

## Endpoints

- `GET /` — Health check & API info
- `POST /predict` — Predict flight price (JSON body)
- `GET /ui` — Web UI for predictions
- `GET /metrics` — Prediction metrics

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for Swagger UI.

## Docker

```bash
docker build -t voyage-analytics-api .
docker run -p 8000:8000 voyage-analytics-api
```
