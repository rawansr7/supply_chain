import pandas as pd
from analysis.lstm import downcast, preprocess_data, train, get_predictions
import numpy as np

# def preprocess_data(data):
#     df["sale_date"] = pd.to_datetime(df["sale_date"])
#     df = df.sort_values(by=["sale_date"])

#     # Feature engineering
#     df["year"] = df["sale_date"].dt.year
#     df["month"] = df["sale_date"].dt.month
#     df["week"] = df["sale_date"].dt.isocalendar().week
#     df["day"] = df["sale_date"].dt.day
#     df = df.drop(columns=["sale_date"])
#     return df


def forecast_next_month(data):
    df = pd.DataFrame(data)
    products = df["item_id"].to_list()
    stores = df["store_id"].to_list()

    df = downcast(df)
    X_train, y_train, X_val, y_val, scaler = preprocess_data(df, False)

    model = train(  # use best parameters
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=0.01,
        batch_size=50,
    )
    predictions = get_predictions(y_val, model)
    predictions = scaler.inverse_transform(predictions).tolist()
    predictions = predictions.sum(axis=0)

    forecast_results = []
    for i in range(len(predictions)):
        store = stores[i]
        product = products[i]
        forecasted_quantity = round(predictions[i])
        forecast_results.append({"store": store, "product": product, "forecasted_quantity": forecasted_quantity})

    return forecast_results
