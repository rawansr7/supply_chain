from datetime import timedelta
from analysis.lstm import downcast, preprocess_data, train, get_predictions_for_unseen_data
import pandas as pd


def forecast_next_month(df):
    df = downcast(df)

    end_date = df["date"].max()

    X_train, y_train, X_val, y_val, scaler, items_and_stores_ids = preprocess_data(df, False)
    model, min_val_rmse = train(  # use best parameters
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=0.01,
        batch_size=50,
    )
    predictions = get_predictions_for_unseen_data(y_val, model)
    predictions = scaler.inverse_transform(predictions)
    predictions[predictions < 0] = 0

    forecast_results = []
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            product, store = items_and_stores_ids[j]
            forecasted_sold = round(predictions[i][j])
            forecasted_date = end_date + timedelta(days=i + 1)
            forecast_results.append(
                {
                    "store_id": store,
                    "item_id": product,
                    "forecasted_sold": forecasted_sold,
                    "forecasted_date": forecasted_date,
                }
            )
    forecast_results = pd.DataFrame.from_records(forecast_results)
    return forecast_results
