from fastapi import FastAPI
from pydantic import BaseModel
import joblib as jb
from typing import Optional
import pandas as pd

app = FastAPI()

model = jb.load("model.pkl")
encoders = jb.load("label_encoders.pkl")

class get_details(BaseModel):
    Education: str
    JoiningYear: int
    City: Optional[str] = "Bangalore"
    PaymentTier: Optional[int] = None
    Age: int
    Gender: Optional[str] = "Male"
    EverBenched: Optional[str] = "No"
    ExperienceInCurrentDomain: int

@app.post("/predict")
def predict(data:get_details):

    if data.City is not None:
        city_val = data.City
    else:
        city_val = "Bangalore"

    if data.PaymentTier is not None:
        Pay_val = data.PaymentTier
    else:
        Pay_val = 2

    if data.Gender is not None:
        Gender_val = data.Gender
    else:
        Gender_val = "Male"

    if data.EverBenched is not None:
        EverBenched_val = data.EverBenched
    else:
        EverBenched_val = "No"

    encoded_education = encoders["Education"].transform([data.Education])[0]
    encoded_city = encoders["City"].transform([city_val])[0]
    encode_gender = encoders["Gender"].transform([Gender_val])[0]
    encode_EverBenched = encoders["EverBenched"].transform([EverBenched_val])[0]
    
    features=[[
        encoded_education,
        data.JoiningYear,
        encoded_city ,
        Pay_val,
        data.Age,
        encode_gender,
        encode_EverBenched,
        data.ExperienceInCurrentDomain
    ]]

    prediction = float(model.predict_proba(features)[0][1])

    if prediction >=0.51:
        risk_label = "High Risk of Leaving"
    elif prediction < 0.49:
        risk_label = "Low Risk of Leaving"
    else:
        risk_label = "Uncertain / 50-50 Chance"


    return {
        "leave_probability": round(float(prediction), 4),
        "risk_level": risk_label,
        "prediction_code": 1 if prediction > 0.50 else 0
    }