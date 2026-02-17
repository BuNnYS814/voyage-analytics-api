from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, constr
from src.services.predict_service import predict
from src.monitoring.logger import log_prediction, get_metrics

app = FastAPI(title="Voyage Analytics Pro API")


class FlightFeatures(BaseModel):
    travelCode: int = Field(..., ge=0)
    userCode: int = Field(..., ge=0)
    origin: constr(min_length=1)
    destination: constr(min_length=1)
    flightType: constr(min_length=1)
    time: float = Field(..., gt=0)
    distance: float = Field(..., gt=0)
    agency: constr(min_length=1)
    date: constr(min_length=1)  # accepts multiple formats, parsed server-side


@app.get("/")
def read_root():
    return {
        "message": "Voyage Analytics Pro API is running",
        "predict_endpoint": "/predict",
        "ui": "/ui",
        "metrics": "/metrics",
    }


@app.get("/metrics")
def read_metrics():
    """Expose simple prediction metrics for monitoring."""
    return get_metrics()


@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    """Simple HTML frontend for making predictions."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Voyage Analytics Pro - Flight Price Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; background: #050816; color: #f9fafb; margin: 0; padding: 0; }
            .container { max-width: 900px; margin: 40px auto; padding: 24px; background: #111827; border-radius: 12px; box-shadow: 0 20px 40px rgba(0,0,0,0.6); }
            h1 { margin-bottom: 8px; }
            p { margin-top: 0; color: #9ca3af; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 24px; }
            label { display: block; font-size: 0.9rem; margin-bottom: 4px; color: #d1d5db; }
            input, select { width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid #4b5563; background: #020617; color: #f9fafb; }
            input:focus, select:focus { outline: none; border-color: #6366f1; box-shadow: 0 0 0 1px #6366f1; }
            button { margin-top: 20px; padding: 10px 18px; border-radius: 999px; border: none; background: linear-gradient(135deg, #6366f1, #ec4899); color: white; font-weight: 600; cursor: pointer; }
            button:hover { opacity: 0.95; transform: translateY(-1px); }
            .result { margin-top: 24px; padding: 16px; border-radius: 10px; background: #020617; border: 1px solid #374151; }
            .result strong { font-size: 1.2rem; }
            .error { color: #fecaca; margin-top: 8px; }
            small { color: #6b7280; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Voyage Analytics Pro</h1>
            <p>Estimate a flight price using the trained XGBoost model.</p>

            <form id="predict-form">
                <div class="grid">
                    <div>
                        <label>Travel Code</label>
                        <input type="number" id="travelCode" value="0" min="0" required />
                    </div>
                    <div>
                        <label>User Code</label>
                        <input type="number" id="userCode" value="0" min="0" required />
                    </div>
                    <div>
                        <label>Origin</label>
                        <input type="text" id="origin" value="Recife (PE)" required />
                    </div>
                    <div>
                        <label>Destination</label>
                        <input type="text" id="destination" value="Florianopolis (SC)" required />
                    </div>
                    <div>
                        <label>Flight Type</label>
                        <select id="flightType">
                            <option>economic</option>
                            <option>firstClass</option>
                            <option>premium</option>
                        </select>
                    </div>
                    <div>
                        <label>Time (hours)</label>
                        <input type="number" step="0.01" id="time" value="1.76" required />
                    </div>
                    <div>
                        <label>Distance (km)</label>
                        <input type="number" step="0.01" id="distance" value="676.53" required />
                    </div>
                    <div>
                        <label>Agency</label>
                        <select id="agency">
                            <option>FlyingDrops</option>
                            <option>CloudFy</option>
                            <option>Rainbow</option>
                        </select>
                    </div>
                    <div>
                        <label>Date</label>
                        <input type="date" id="date" />
                        <small>Defaults to today if left empty.</small>
                    </div>
                </div>

                <button type="submit">Predict Price</button>
                <div id="error" class="error"></div>
            </form>

            <div id="result" class="result" style="display:none;">
                <strong>Predicted price: <span id="price"></span></strong>
            </div>
        </div>

        <script>
            const form = document.getElementById('predict-form');
            const resultBox = document.getElementById('result');
            const priceSpan = document.getElementById('price');
            const errorBox = document.getElementById('error');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                errorBox.textContent = '';
                resultBox.style.display = 'none';

                const today = new Date().toISOString().split('T')[0];

                const payload = {
                    travelCode: parseInt(document.getElementById('travelCode').value || '0'),
                    userCode: parseInt(document.getElementById('userCode').value || '0'),
                    origin: document.getElementById('origin').value,
                    destination: document.getElementById('destination').value,
                    flightType: document.getElementById('flightType').value,
                    time: parseFloat(document.getElementById('time').value),
                    distance: parseFloat(document.getElementById('distance').value),
                    agency: document.getElementById('agency').value,
                    date: document.getElementById('date').value || today
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    const data = await res.json();
                    if (!res.ok) {
                        throw new Error(data.detail || 'Prediction failed');
                    }

                    priceSpan.textContent = data.predicted_price.toFixed(2);
                    resultBox.style.display = 'block';
                } catch (err) {
                    errorBox.textContent = err.message;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/predict")
def get_prediction(data: FlightFeatures):
    raw_features = {
        "travelCode": data.travelCode,
        "userCode": data.userCode,
        "from": data.origin,
        "to": data.destination,
        "flightType": data.flightType,
        "time": data.time,
        "distance": data.distance,
        "agency": data.agency,
        "date": data.date,
    }
    try:
        result = predict(raw_features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    log_prediction(raw_features, result)
    return {"predicted_price": result}
