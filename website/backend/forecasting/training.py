import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def preprocess_data(data):
    df = pd.DataFrame(data)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values(by=["sale_date"])

    # Feature engineering
    df["year"] = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month
    df["week"] = df["sale_date"].dt.isocalendar().week
    df["day"] = df["sale_date"].dt.day
    df = df.drop(columns=["sale_date"])
    return df


def train_models(df):
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["store", "product"])

    # Split data
    X = df.drop(columns=["quantity"])
    y = df["quantity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "XGBoost": XGBRegressor(random_state=42),
    }

    best_model = None
    best_mse = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        if mse < best_mse:
            best_model = model
            best_mse = mse

    return best_model


def forecast_next_week(data):

    df = preprocess_data(data)
    best_model = train_models(df)

    # Forecast for the next week
    stores = df["store"].unique()
    products = df["product"].unique()
    forecast_results = []

    for store in stores:
        for product in products:
            row = {
                "store": store,
                "product": product,
                "year": 2024,
                "month": 12,
                "week": 52,
                "day": 1,
            }
            for col in df.columns:
                if col.startswith("store_"):
                    row[col] = 1 if col == f"store_{store}" else 0
                elif col.startswith("product_"):
                    row[col] = 1 if col == f"product_{product}" else 0
            row_df = pd.DataFrame([row])
            forecasted_quantity = int(best_model.predict(row_df)[0])

            forecast_results.append(
                {
                    "store": store,
                    "product": product,
                    "forecasted_quantity": forecasted_quantity,
                }
            )

    return forecast_results
