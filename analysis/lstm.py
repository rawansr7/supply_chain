import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import json

tf.random.set_seed(51)
np.random.seed(51)


def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, t in enumerate(types):
        if "int" in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif "float" in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == object:
            if cols[i] == "date":
                df[cols[i]] = pd.to_datetime(df[cols[i]], format="%Y-%m-%d")
            else:
                df[cols[i]] = df[cols[i]].astype("category")
    return df


def preprocess_data(df, get_test_data):
    # Preprocess: remove id, item_id, dept_id, cat_id, store_id, state_id columns
    print(df)
    df["date"] = pd.to_datetime(df["date"])

    # Create a complete date range from the minimum to the maximum date
    all_dates = pd.date_range(start=df["date"].min(), end=df["date"].max())

    # Pivot the DataFrame using 'item_id' and 'store_id' as the index
    pivoted_df = df.pivot(index=["item_id", "store_id"], columns="date", values="sold")

    # Reindex the columns to include all dates, filling missing dates with 0
    pivoted_df = pivoted_df.reindex(columns=all_dates, fill_value=0)

    # Reset the index and remove axis names
    pivoted_df = pivoted_df.reset_index()
    pivoted_df.columns.name = None

    # Replace dates by integers (1 as the start date)
    date_columns = pivoted_df.columns[2:]  # Exclude 'item_id' and 'store_id'
    date_mapping = {date: i + 1 for i, date in enumerate(date_columns)}
    pivoted_df = pivoted_df.rename(columns=date_mapping)

    sales = pivoted_df.T
    items_and_stores_ids = list(zip(sales[0], sales[1]))

    sales = sales[2:]
    total_timesteps = len(sales)
    sc = MinMaxScaler(feature_range=(0, 1))
    train_sales_scaled = sc.fit_transform(sales)
    timesteps = 28  # use the last 28 days to predict the next day's sales
    X = []
    y = []

    for i in range(timesteps, total_timesteps):
        X.append(train_sales_scaled[i - 28 : i])  # noqa
        y.append(train_sales_scaled[i])
    # Convert to np array to be able to feed the LSTM model
    X = np.array(X)
    y = np.array(y)

    if get_test_data:
        X_train, X_val, X_test = X[:-56], X[-56:-28], X[-28:]
        y_train, y_val, y_test = y[:-56], y[-56:-28], y[-28:]
        return X_train, y_train, X_val, y_val, X_test, y_test, sc, items_and_stores_ids
    else:
        X_train, X_val = X[:-28], X[-28:]
        y_train, y_val = y[:-28], y[-28:]
        return X_train, y_train, X_val, y_val, sc, items_and_stores_ids


def train(X_train, y_train, X_val, y_val, learning_rate, batch_size):
    n_timesteps = X_train.shape[1]
    n_products = X_train.shape[2]

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=7,
                strides=1,
                padding="causal",
                activation="relu",
                input_shape=(n_timesteps, n_products),
            ),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=7,
                strides=1,
                activation="relu",
                padding="causal",
            ),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(n_products),
        ]
    )

    opt_adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss="mse",
        optimizer=opt_adam,
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error",  # Metric to monitor
        patience=5,  # Number of epochs with no improvement to wait
        restore_best_weights=True,  # Restore weights from the best epoch
        mode="min",
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=batch_size,
        callbacks=[early_stopping],
    )
    min_val_rmse = min(history.history["val_root_mean_squared_error"])
    return model, min_val_rmse


def get_predictions(X_test, model):
    predictions = []
    for i in range(X_test.shape[0]):
        prediction = model.predict(np.expand_dims(X_test[i], axis=0)).tolist()[0]
        predictions.append(prediction)
    predictions = np.array(predictions)
    return predictions


def get_predictions_for_unseen_data(y_val, model):
    predictions = []
    num_days = 28
    for i in range(num_days):
        prediction = model.predict(np.expand_dims(y_val, axis=0)).tolist()[0]
        predictions.append(prediction)
        y_val = np.append(y_val[1:], np.array([prediction]), axis=0)
    predictions = np.array(predictions)
    return predictions


if __name__ == "__main__":
    sales = pd.read_csv("data/reduced_m5_dataset.csv")
    sales = downcast(sales)
    X_train, y_train, X_val, y_val, X_test, y_test, sc, items_and_stores_ids = preprocess_data(sales, True)

    all_params = []
    best_param = None
    best_score = 9999
    for learning_rate in [0.001, 0.005, 0.01][-1:]:
        for batch_size in [50, 100, 200][-1:]:
            model, min_val_rmse = train(
                X_train,
                y_train,
                X_val,
                y_val,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            predictions = get_predictions(X_test, model)
            predictions_unscaled = sc.inverse_transform(predictions)
            y_test_unscaled = sc.inverse_transform(y_test)
            mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
            root_mse = root_mean_squared_error(y_test_unscaled, predictions_unscaled)
            all_params.append(
                {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "mae": mae,
                    "rmse": root_mse,
                    "val_rmse": min_val_rmse,
                }
            )
            if min_val_rmse < best_score:
                best_score = min_val_rmse
                best_param = all_params[-1]

    json.dump(all_params, open("lstm_ttt.json", "w"))
    print(best_param)
