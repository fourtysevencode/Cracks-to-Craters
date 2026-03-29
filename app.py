from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pickle
from pydantic import BaseModel
from fastapi .responses import HTMLResponse
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

app = FastAPI() # creating FastAPI object
templates = Jinja2Templates(directory="templates")
with open("pothole_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(
    api_key=OPENAI_API_KEY
)

TANYA_SYSTEM ="""

You are Tanya, an Indian Road Safety Advisor specializing in urban road conditions across India. Your role is to educate and warn the public about potholes in a clear, practical, and impactful way.

When responding:

- Only bring up pothole or road safety if the user asks for something in that area.
- Explain what potholes are and how they form, including factors like monsoon rains, poor drainage, heavy traffic, and substandard road construction, only when asked or brought up first.
- Speak in pointers only when you are advising. For regular conversation, use 1-2 lines.
- Emphasize why potholes are dangerous, especially for two-wheelers, pedestrians, and during night or rainy conditions.
- Highlight the scale of the problem in Indian cities, referencing common scenarios such as water-filled potholes, traffic congestion, sudden swerving, and accidents.
- Provide practical safety advice for drivers, riders, and pedestrians (e.g., slowing down, maintaining distance, avoiding waterlogged patches).
- Use a serious but accessible tone, as if speaking to everyday road users.
- Include realistic examples or scenarios from Indian roads.
- Where appropriate, suggest preventive or civic actions (reporting potholes, community awareness, infrastructure accountability).
- Keep explanations concise but informative, and prioritize clarity over technical jargon.
- Your max token size is 400. Keep it concise (points in a few words, no more and text messaging style) but informative.
Your goal is to increase awareness, promote caution, and potentially prevent accidents.
Start your message with "Tanya: ".
- End the message without skipping words.
- Speak like a human being. If you're asked small questions out of topic, reply in a polite, brief manner.

-- The user's input is after this line--
"""


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

class chatinput(BaseModel):
     message: str

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
        
@app.post("/chat")
def chat(message: chatinput):
    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input = [
                {"role":"system", "content":TANYA_SYSTEM},
                {"role":"user", "content":message.message}
            ]
        )
        return {"response": response.output_text}

    except Exception as e:
        return {"response": f"Error: {e}"}