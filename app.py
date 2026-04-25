from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pickle
from pydantic import BaseModel
from fastapi .responses import HTMLResponse
import numpy as np

app = FastAPI() # creating FastAPI object
templates = Jinja2Templates(directory="templates")
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
    soil_type: str  # if empty, returns gravel so it doesnt crash

@app.get("/", response_class=HTMLResponse) # HTML response, not json
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict")
def predict(data: InputData):
        try:
            soil = data.soil_type 

            features = {col: 0.0 for col in FEATURE_COLUMNS}
            for key in ["avg_rainfall_mm", "traffic_volume_vph", "pavement_age_yrs", "last_repair_yrs"]:
                features[key] = float(getattr(data, key,0))

            soil_col = f"soil_type_{soil.lower()}"
            if soil_col in features:
                features[soil_col] = 1.0

            X = np.array([[features[c] for c in FEATURE_COLUMNS]])
            prediction = model.predict(X)[0]

            return {"prediction": (prediction)}
         
        except Exception as e:
            return {"prediction": f"Error: {e}"}
        
