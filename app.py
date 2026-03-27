from fastapi import FastAPI, Request, Body
from fastapi.templating import Jinja2Templates
import pickle
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
with open("pothole_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_COLUMNS = [
    "avg_rainfall_mm",
    "traffic_volume_vph",
    "pavement_age_yrs",
    "last_repair_yrs",
    "soil_type_clay",
    "soil_type_gravel",
    "soil_type_sandy"
]

class InputData(BaseModel):
    avg_rainfall_mm: float
    traffic_volume_vph: float
    pavement_age_yrs: float
    last_repair_yrs: float
    soil_type: str


# HOME PAGE
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# PREDICTION ROUTE (UNCHANGED)
@app.post("/predict")
def predict(data: InputData):
    try:
        soil = data.soil_type

        features = {col: 0.0 for col in FEATURE_COLUMNS}

        for key in ["avg_rainfall_mm", "traffic_volume_vph", "pavement_age_yrs", "last_repair_yrs"]:
            features[key] = float(getattr(data, key, 0))

        soil_col = f"soil_type_{soil.lower()}"
        if soil_col in features:
            features[soil_col] = 1.0

        X = np.array([[features[c] for c in FEATURE_COLUMNS]])
        prediction = model.predict(X)[0]

        return {"prediction": prediction}

    except Exception as e:
        return {"prediction": f"Error: {e}"}


# ------------------- CHATBOT STARTS HERE -------------------

def generate_advice(prediction):
    if prediction == 1:
        return "High Risk: Avoid this road, especially in rain. Slow down and maintain distance."
    elif prediction == 0:
        return "Low Risk: Road seems safe, but stay alert."
    else:
        return "Moderate Risk: Drive carefully and avoid waterlogged areas."


@app.post("/chat")
async def chat(message: dict = Body(...)):
    user_input = message.get("message", "").lower()

    try:
        rainfall = None
        traffic = None

        words = user_input.split()

        # Extract rainfall
        for word in words:
            if word.isdigit():
                rainfall = float(word)

        # Detect traffic level
        if "high traffic" in user_input:
            traffic = 2000
        elif "medium traffic" in user_input:
            traffic = 1000
        elif "low traffic" in user_input:
            traffic = 300

        # If valid inputs → run model
        if rainfall and traffic:
            X = np.array([[rainfall, traffic, 5, 2, 0, 1, 0]])
            prediction = model.predict(X)[0]

            advice = generate_advice(prediction)

            return {
                "response": f"Tanya:\nPrediction: {prediction}\n{advice}"
            }

        # If inputs missing
        return {
            "response": "Tanya: Please enter rainfall (number) and traffic (high/medium/low). Example: rainfall 60 high traffic"
        }

    except Exception as e:
        return {"response": f"Tanya: Error - {e}"}
