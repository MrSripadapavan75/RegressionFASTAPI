
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# 1. Load the dataset
df = pd.read_csv("House.csv")

# 2. Select exactly the features used by the HTML form
FEATURES = [
    "LotArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "1stFlrSF",
    "GarageCars"
]

TARGET = "SalePrice"

# 3. Check that the required dataset columns exist
required_columns = FEATURES + [TARGET]
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"These columns are missing from House.csv: {missing_columns}"
    )

# 4. Create input features and target
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Remove rows where the target price is missing
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

print("Input features:", FEATURES)
print("Input data shape:", X.shape)
print("Target data shape:", y.shape)

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Create the complete pipeline
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                objective="reg:squarederror"
            )
        )
    ]
)

# 7. Train the pipeline
pipeline.fit(X_train, y_train)

print("\nModel training completed.")

# 8. Evaluate the model
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# 9. Save the complete pipeline
joblib.dump(pipeline, "house_price_model.pkl")

print("\nSaved: house_price_model.pkl")

# 10. Test that the saved model can be loaded
loaded_model = joblib.load("house_price_model.pkl")

sample_input = X_test.iloc[[0]]
sample_prediction = loaded_model.predict(sample_input)

print("\nSample input:")
print(sample_input)

print("\nSample predicted price:", round(float(sample_prediction[0]), 2))
print("Saved model loaded and tested successfully.")
