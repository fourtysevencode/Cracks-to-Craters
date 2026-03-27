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


# ---------------- HOME ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------- PREDICT ----------------
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


# ---------------- CHATBOT ----------------

def generate_advice(prediction):
    if prediction == 1:
        return "High Risk: Avoid this road, especially during rain. Reduce speed and maintain distance."
    elif prediction == 0:
        return "Low Risk: Road seems relatively safe, but stay alert."
    else:
        return "Moderate Risk: Drive carefully and avoid waterlogged patches."


@app.post("/chat")
async def chat(message: dict = Body(...)):
    user_input = message.get("message", "").lower()

    rainfall = None
    traffic = None

    # Extract rainfall (handles decimals like 80.5)
    for word in user_input.split():
        try:
            rainfall = float(word)
        except:
            pass

    # Detect traffic level
    if "high traffic" in user_input:
        traffic = 2000
    elif "medium traffic" in user_input:
        traffic = 1000
    elif "low traffic" in user_input:
        traffic = 300

    # Run model if valid input
    if rainfall is not None and traffic is not None:
        X = np.array([[rainfall, traffic, 5, 2, 0, 1, 0]])
        prediction = model.predict(X)[0]

        # Better risk labeling
        if prediction == 1:
            risk_label = "High Risk"
        elif prediction == 0:
            risk_label = "Low Risk"
        else:
            risk_label = "Moderate Risk"

        return {
            "response": f"Tanya:\nRisk: {risk_label}\n{generate_advice(prediction)}"
        }

    # If input incomplete
    return {
        "response": "Tanya: Please enter rainfall (number) and traffic (high/medium/low).\nExample: rainfall 60 high traffic"
    }
