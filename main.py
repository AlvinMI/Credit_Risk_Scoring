from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title = 'Credit Risk Scoring API',
    description = 'Production-grade API for predicting credit default probability using XGBoost',
    version = '1.0.0'
)

MODEL_PATH = ('models/xgb_credit_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("Model XGBoost berhasil di-load!")
except Exception as e :
    print(f'Error loading model: {e}')

class CreditRequest(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., example=0.5)
    age: int = Field(..., example=45)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., example=0)
    DebtRatio: float = Field(..., example=0.3)
    MonthlyIncome: float = Field(..., example=5000.0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., example=5)
    NumberOfTimes90DaysLate: int = Field(..., example=0)
    NumberRealEstateLoansOrLines: int = Field(..., example=1)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., example=0)
    NumberOfDependents: int = Field(..., example=2.0)
    
@app.get('/')
def read_root():
    return{'status': 'online', 'message': 'Credit Risk Scoring API is running!'}

@app.post('/predict')
def predict_credit_risk(request: CreditRequest):
    try:
        data_dict = request.dict()
        
        total_past_due = (
            data_dict['NumberOfTime30_59DaysPastDueNotWorse'] +
            data_dict['NumberOfTime60_89DaysPastDueNotWorse'] +
            data_dict['NumberOfTimes90DaysLate']
        )
        monthly_debt = data_dict['MonthlyIncome'] * data_dict['DebtRatio']
        
        input_data = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': data_dict['RevolvingUtilizationOfUnsecuredLines'],
            'age': data_dict['age'],
            'NumberOfTime30-59DaysPastDueNotWorse': data_dict['NumberOfTime30_59DaysPastDueNotWorse'],
            'DebtRatio': data_dict['DebtRatio'],
            'MonthlyIncome': data_dict['MonthlyIncome'],
            'NumberOfOpenCreditLinesAndLoans': data_dict['NumberOfOpenCreditLinesAndLoans'],
            'NumberOfTimes90DaysLate': data_dict['NumberOfTimes90DaysLate'],
            'NumberRealEstateLoansOrLines': data_dict['NumberRealEstateLoansOrLines'],
            'NumberOfTime60-89DaysPastDueNotWorse': data_dict['NumberOfTime60_89DaysPastDueNotWorse'],
            'NumberOfDependents': data_dict['NumberOfDependents'],
            'TotalPastDue': total_past_due,
            'MonthlyDebt': monthly_debt
        }])
        
        probability_default = float(model.predict_proba(input_data)[:,1][0])
        
        status = 'REJECTED' if probability_default >= 0.35 else 'APPROVED'
        
        return{
            'status': status,
            'default_probability': round(probability_default, 4),
            'risk_score_percent': f'{round(probability_default * 100, 2)}%',
            'decision': 'Permohonan Kredit Ditolak (Resiko Tinggi)' if status == 'REJECTED' else 'Permohonan Kredit Disetujui'
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))