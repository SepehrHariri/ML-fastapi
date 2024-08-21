from fastapi import FastAPI
from pydantic import BaseModel
import pickle

import uvicorn
import joblib

# from app.model.model import predict_pipeline
# from app.model.model import __version__ as model_version


# from typing import Union


import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import explained_variance_score

# # define model info
# model_name = 'randomforest'
# model_path = "app/model/" + model_name

# # load model
# model = RandomForestRegressor(model_path)

# #create distance metric object
# mse = mean_squared_error(out_test, out_pred)
# rmse = mse**0.5



app = FastAPI(debug=False)
# app = FastAPI(debug=True)


pickle_in = open("trained_pipeline-0.1.0.pkl", "rb")
model=pickle.load(pickle_in)



@app.get("/")
def home():
    # return {"health_check": "OK", "model_version": model_version}
    return {"text": "Metalization Prediction"}


# @app.post("/predict", response_model=PredictionOut)
# def predict(payload: Meta):
#     language = predict_pipeline(payload.text)
#     return {"language": language}

@app.post('/predict')
def predict_md(
        AITAA181ValueY:float,	AITAA331ValueY:float,	AITAA332ValueY:float,	AITAA311ValueY:float,	AITAA322BValueY:float,
        AITAA322DValueY:float,	AITAA321ValueY:float,	AITAA322AValueY:float,	AITAD181ValueY:float,	AITAD182ValueY:float,
    	FTAA29ValueY:float,	FTAA82ValueY:float,	FYAA39ValueY:float,	FIAG281ValueY:float,	FIAD58ValueY:float,	FTAD24ValueY:float,
        FTAA19ValueY:float,	FTAA33ValueY:float,	FICAE14ValueY:float,	FICAE16ValueY:float,	FICAE09ValueY:float,	PDIAA46ValueY:float,
    	PDTAD541ValueY:float,	PDTAD542ValueY:float,	PTAA19ValueY:float,	PTAB081ValueY:float,	PTAB082ValueY:float,	PDIAA01ValueY:float,
    	PTAC071ValueY:float,	PTAC072ValueY:float,	PTAD15ValueY:float,	TTAA521ValueY:float,	TTAA522ValueY:float,	TTAA5213ValueY:float,
        TTAA5216ValueY:float,	TTAA523ValueY:float,	TTAA524ValueY:float,	TTAB191ValueY:float,	TTAA251ValueY:float,	TTAA252ValueY:float,
    	TTAA841ValueY:float,	TTAB401BValueY:float,	TTAB402BValueY:float,	TTAB403BValueY:float,	TTAB404BValueY:float,
    	TTAA341ValueY:float,	TTAA342ValueY:float,	TTAA35ValueY:float,	TTAH80ValueY:float,	TTAD27ValueY:float,	TTAA843ValueY:float,
    	TTAA844ValueY:float,	TTAB221ValueY:float,	TTAB222ValueY:float,
    ):

    # data = data.dict()
    # FYAA39ValueY=data['FY-AA39 ValueY']
    # MD=data['MD%']

    makeprediction = model.predict([[
        AITAA181ValueY,	AITAA331ValueY,	AITAA332ValueY,	AITAA311ValueY,	AITAA322BValueY,
        AITAA322DValueY,	AITAA321ValueY,	AITAA322AValueY,	AITAD181ValueY,	AITAD182ValueY,
    	FTAA29ValueY,	FTAA82ValueY,	FYAA39ValueY,	FIAG281ValueY,	FIAD58ValueY,	FTAD24ValueY,
        FTAA19ValueY,	FTAA33ValueY,	FICAE14ValueY,	FICAE16ValueY,	FICAE09ValueY,	PDIAA46ValueY,
    	PDTAD541ValueY,	PDTAD542ValueY,	PTAA19ValueY,	PTAB081ValueY,	PTAB082ValueY,	PDIAA01ValueY,
    	PTAC071ValueY,	PTAC072ValueY,	PTAD15ValueY,	TTAA521ValueY,	TTAA522ValueY,	TTAA5213ValueY,
        TTAA5216ValueY,	TTAA523ValueY,	TTAA524ValueY,	TTAB191ValueY,	TTAA251ValueY,	TTAA252ValueY,
    	TTAA841ValueY,	TTAB401BValueY,	TTAB402BValueY,	TTAB403BValueY,	TTAB404BValueY,
    	TTAA341ValueY,	TTAA342ValueY,	TTAA35ValueY,	TTAH80ValueY,	TTAD27ValueY,	TTAA843ValueY,
    	TTAA844ValueY,	TTAB221ValueY,	TTAB222ValueY,
        ]])
    
    output = round(makeprediction[0],2)
    return {
        'Your Metalization is: {}'.format(output) 
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#  uvicorn main:app --reload
