import pandas as pd
from datetime import timedelta
from analysis.lstm import downcast, preprocess_data, train, get_predictions_for_unseen_data


def forecast_next_month(data):
    df = pd.DataFrame(data)
    df = downcast(df)

    end_date = df["date"].max()

    X_train, y_train, X_val, y_val, scaler, items_and_stores_ids = preprocess_data(df, False)

    model = train(  # use best parameters
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
    for day in range(1, predictions.shape[0] + 1):
        for i in range(predictions[day]):
            product, store = items_and_stores_ids[i]
            forecasted_sold = round(predictions[day][i])
            forecasted_date = end_date + timedelta(days=day)
            forecast_results.append(
                {
                    "store_id": store,
                    "item_id": product,
                    "forecasted_sold": forecasted_sold,
                    "forecasted_date": forecasted_date,
                }
            )

    return forecast_results
