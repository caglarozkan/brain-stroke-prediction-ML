from fastapi import FastAPI,Request,Response
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
#uvicorn DOSYA_ADI:FastAPI_DEGISKENI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm sitelerden gelen isteklere izin ver (Geliştirme aşaması için)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

with open("model_complete.pkl", "rb") as f:
    saved_model = pickle.load(f)

    model=saved_model["model"]
    scaler=saved_model["scaler"]
    threshold=saved_model["threshold"]
# Load trained model, preprocessor and decision threshold

class BrainStrokeFeatures(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.get("/",response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/predict")
async def predict_test(features: BrainStrokeFeatures):
    input_data=pd.DataFrame([features.model_dump()])
    print(input_data)

    input_processed = scaler.transform(input_data)
    probability = model.predict_proba(input_processed)[:,1][0]
    prediction = 1 if probability >= threshold else 0

    return {
        "stroke_probability": round(float(probability), 4,),
        "threshold_used":threshold,
        "prediction":int(prediction),
        "prediction_label":"Risk Var" if prediction == 1 else "Risk Düşük"
    }