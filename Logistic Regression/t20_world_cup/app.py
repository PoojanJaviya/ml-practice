from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware




class MatchInput(BaseModel):
    Venue : str
    Team_A : str
    Team_B : str
    Stage : str
    Team_A_Ranking : int
    Team_B_Ranking : int
    Team_A_Form : float
    Team_B_Form : float
    HeadToHead_A_Wins : int
    HeadToHead_B_Wins : int

    Venue_HomeAdvantage_A : int = 0
    Venue_HomeAdvantage_B : int = 0

    Pitch_Type : str
    Avg_T20_Score_Venue : int
    Toss_Winner : str
    Toss_Decision : str
    Team_A_Tech_Index : float
    Team_B_Tech_Index : float
    Match_Total : int

CATEGORICAL_COLS = [
    "Venue",
    "Team_A",
    "Team_B",
    "Stage",
    "Pitch_Type",
    "Toss_Winner",
    "Toss_Decision"
]

NUMERIC_COLS = [
    "Team_A_Ranking",
    "Team_B_Ranking",
    "Team_A_Form",
    "Team_B_Form",
    "HeadToHead_A_Wins",
    "HeadToHead_B_Wins",
    "Venue_HomeAdvantage_A",
    "Venue_HomeAdvantage_B",
    "Avg_T20_Score_Venue",
    "Team_A_Tech_Index",
    "Team_B_Tech_Index",
    "Match_Total"
]

def preprocess_input(data, onehot_encoder):
    df = pd.DataFrame([data])

    # Encode ONLY categorical columns
    cat_encoded = onehot_encoder.transform(df[CATEGORICAL_COLS])

    # Numeric columns stay as-is
    num_data = df[NUMERIC_COLS].values

    # Combine both
    X = np.hstack([num_data, cat_encoded])

    return X


model = joblib.load('t20_wc_26.pkl')
label_encoders = joblib.load('label_encoder.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
def predict(inp : MatchInput,):
    processed = preprocess_input(inp.dict(), onehot_encoder)

    prediction = model.predict(processed)[0]
    winner = label_encoders.inverse_transform([prediction])[0]

    if winner == "Team_A":
        winner = inp.dict()["Team_A"]
    else:
        winner = inp.dict()["Team_B"]

    return {
        "predicted_winner" : winner
    }
