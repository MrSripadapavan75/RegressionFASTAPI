
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Application paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
TEMPLATE_DIR = BASE_DIR / "templates"


# --------------------------------------------------
# 2. Load the trained pipeline
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "LotArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "1stFlrSF",
    "GarageCars"
]


# --------------------------------------------------
# 3. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house sale prices using XGBoost",
    version="1.0.0"
)

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)


# --------------------------------------------------
# 4. Define and validate the input data
# --------------------------------------------------

class HouseFeatures(BaseModel):
    LotArea: float = Field(gt=0)
    OverallQual: int = Field(ge=1, le=10)
    YearBuilt: int = Field(ge=1800, le=2026)
    TotalBsmtSF: float = Field(ge=0)
    first_flr_sf: float = Field(gt=0, alias="1stFlrSF")
    GarageCars: int = Field(ge=0, le=5)


# --------------------------------------------------
# 5. Display the user interface
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_house_price(data: HouseFeatures):

    try:
        input_data = pd.DataFrame(
            [{
                "LotArea": data.LotArea,
                "OverallQual": data.OverallQual,
                "YearBuilt": data.YearBuilt,
                "TotalBsmtSF": data.TotalBsmtSF,
                "1stFlrSF": data.first_flr_sf,
                "GarageCars": data.GarageCars
            }],
            columns=FEATURES
        )

        prediction = model.predict(input_data)

        predicted_price = float(prediction[0])

        return {
            "predicted_price": round(predicted_price, 2),
            "message": "Prediction generated successfully"
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the application logs."
        ) from exc


# --------------------------------------------------
# 7. Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }
