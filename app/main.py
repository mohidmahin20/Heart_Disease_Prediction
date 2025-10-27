from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()


model = joblib.load('model/heart_model.joblib')

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/info")
async def info():
    return {"model": "Random Forest", "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]}


@app.post("/predict")
async def predict(data: HeartDiseaseInput):
    input_data = [[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]]
    prediction = model.predict(input_data)
    return {"heart_disease": bool(prediction[0])}
