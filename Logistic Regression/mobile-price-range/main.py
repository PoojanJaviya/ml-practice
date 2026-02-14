from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
model = joblib.load("model.pkl")

class PredictInput(BaseModel):
    battery_power :int
    blue : int        
    clock_speed : float    
    dual_sim : int   
    fc : int  
    four_g : int         
    int_memory : int     
    m_dep : float          
    mobile_wt : int
    n_cores : int        
    pc : int              
    px_height : int      
    px_width : int       
    ram : int           
    sc_h  : int          
    sc_w : int            
    talk_time : int       
    three_g : int 
    touch_screen : int   
    wifi : int          

app = FastAPI()

def format_input(data: PredictInput):
    return [[
        data.battery_power,
        data.blue,
        data.clock_speed,
        data.dual_sim,
        data.fc,
        data.four_g,
        data.int_memory,
        data.m_dep,
        data.mobile_wt,
        data.n_cores,
        data.pc,
        data.px_height,
        data.px_width,
        data.ram,
        data.sc_h,
        data.sc_w,
        data.talk_time,
        data.three_g,
        data.touch_screen,
        data.wifi
    ]]


@app.post("/predict")
def predict_price(
    data: PredictInput,
):
    X = format_input(data)
    prediction = model.predict(X)[0]

    return {
        "predicted_price_range": int(prediction)
    }

