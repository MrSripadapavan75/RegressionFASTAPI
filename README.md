# EstateIQ - House Price Prediction

EstateIQ is a house price prediction application built with FastAPI and XGBoost. It provides a browser-based form for entering property details and an API endpoint that returns an estimated sale price.

## Features

- Responsive web interface for entering house features
- House price predictions from a trained XGBoost regression pipeline
- Input validation with Pydantic
- Health check endpoint
- Interactive API documentation
- Docker support

## Tech stack

- Python 3.11
- FastAPI and Uvicorn
- XGBoost, scikit-learn, pandas, and joblib
- Jinja2 templates with Bootstrap and Chart.js

## Project structure

```text
.
├── main.py                         # FastAPI application and prediction endpoint
├── XGBoost_HousePredictionModel.py # Model training and evaluation
├── house_price_model.pkl           # Trained model pipeline used by the API
├── requirements.txt
├── Dockerfile
└── templates/
    └── index.html                  # EstateIQ web interface
```

## Run locally

Use Python 3.11 and run these commands from the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and start the API:

```powershell
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to use the web interface. Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API

### `POST /predict`

Submit the following JSON fields:

| Field | Type | Validation |
|---|---|---|
| `LotArea` | number | Greater than 0 |
| `OverallQual` | integer | 1 to 10 |
| `YearBuilt` | integer | 1800 to 2026 |
| `TotalBsmtSF` | number | 0 or greater |
| `1stFlrSF` | number | Greater than 0 |
| `GarageCars` | integer | 0 to 5 |

Example request:

```json
{
  "LotArea": 8450,
  "OverallQual": 7,
  "YearBuilt": 2003,
  "TotalBsmtSF": 856,
  "1stFlrSF": 856,
  "GarageCars": 2
}
```

Successful responses include `predicted_price` and a message:

```json
{
  "predicted_price": 208500.0,
  "message": "Prediction generated successfully"
}
```

### `GET /health`

Returns the application status and whether the model is loaded.

## Train the model

The model training script expects a `House.csv` file in the project root containing the six feature columns (`LotArea`, `OverallQual`, `YearBuilt`, `TotalBsmtSF`, `1stFlrSF`, and `GarageCars`) and the target column `SalePrice`.

Run training with:

```bash
python XGBoost_HousePredictionModel.py
```

The script splits the data into training and test sets, reports MAE, MSE, RMSE, and R², then saves the trained pipeline to `house_price_model.pkl`. The dataset is excluded from Git by `.gitignore`; provide your own dataset locally if you want to train the model.

## Run with Docker

Build and start the application:

```bash
docker build -t house-prediction-xgboost-api:v1 .
docker run -d --name house-price-prediction -p 8080:8080 house-prediction-xgboost-api:v1
```

Then open [http://localhost:8080](http://localhost:8080).
