import pandas as pd
import numpy as np
import json
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


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


def add_embeddings(df):
    with open("data/embeddings.json", "r") as f:
        embeddings_dict = json.load(f)
    # Embedding size (length of embedding vectors)
    embedding_size = len(next(iter(embeddings_dict.values())))

    # Create new columns for embeddings
    embedding_columns = [f"embedding_{i}" for i in range(embedding_size)]

    embedding_df = pd.DataFrame.from_dict(embeddings_dict, orient="index", columns=embedding_columns, dtype=np.float16)
    embedding_df.reset_index(inplace=True)
    embedding_df.rename(columns={"index": "item_id"}, inplace=True)

    df = pd.merge(df, embedding_df, on="item_id", how="left")
    df["item_id"] = df["item_id"].astype("category")
    return df


def preprocess_data(df, with_text):
    start_date = min(df["date"])
    df["d"] = df["date"].apply(lambda x: (x - start_date).days + 1).astype(np.int16)
    df.drop(columns=["date"], inplace=True)

    if with_text:
        df = add_embeddings(df)

    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, type in enumerate(types):
        if type.name == "category":
            df[cols[i]] = df[cols[i]].cat.codes

    lags = [1, 2, 3, 6, 12, 24, 36]
    max_lag = max(lags)
    for lag in lags:
        df["sold_lag_" + str(lag)] = df.groupby(["item_id", "store_id"], as_index=False)["sold"].shift(lag).astype(np.float16)

    df["rolling_sold_mean"] = df.groupby(["item_id", "store_id"])["sold"].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)

    df = df[df["d"] > max_lag]
    return df


def plot_training_curve(training_metrics, store):

    # Plot training and validation RMSE curves
    plt.figure(figsize=(10, 6))
    plt.plot(training_metrics["training"]["rmse"], label="Training RMSE")
    plt.plot(training_metrics["valid_1"]["rmse"], label="Validation RMSE")
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.title("Training and Validation RMSE")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_" + str(store) + ".png")


def train(df, num_leaves, colsample_bytree):

    train_day = 1886
    val_day = 1914
    final_day = 1942

    test = df[(df["d"] >= val_day) & (df["d"] < final_day)]
    eval_preds = test["sold"]
    eval_true = test["sold"].copy()

    stores = df["store_id"].unique().tolist()

    stores_rmse = []
    for store in stores:
        store_df = df[df["store_id"] == store]

        # Split the data
        X_train = store_df[store_df["d"] < train_day].drop("sold", axis=1)
        y_train = store_df[store_df["d"] < train_day]["sold"]

        X_valid = store_df[(store_df["d"] >= train_day) & (store_df["d"] < val_day)].drop("sold", axis=1)
        y_valid = store_df[(store_df["d"] >= train_day) & (store_df["d"] < val_day)]["sold"]

        X_test = store_df[(store_df["d"] >= val_day) & (store_df["d"] < final_day)].drop("sold", axis=1)
        y_test = store_df[(store_df["d"] >= val_day) & (store_df["d"] < final_day)]["sold"]

        # Train and validate
        model = LGBMRegressor(n_estimators=999999, num_leaves=num_leaves, colsample_bytree=colsample_bytree)

        print(f"*****Prediction for Store: {store}*****")
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric="rmse", callbacks=[early_stopping(5)])
        training_metrics = model.evals_result_
        plot_training_curve(training_metrics, store)

        validation_rmse = training_metrics["valid_1"]["rmse"]
        stores_rmse.append(min(validation_rmse))
        eval_preds[X_test.index] = model.predict(X_test)
        eval_true[X_test.index] = y_test
    val_rmse = np.mean(stores_rmse).tolist()
    mae = mean_absolute_error(eval_true, eval_preds)
    root_mse = root_mean_squared_error(eval_true, eval_preds)

    return root_mse, mae, val_rmse


if __name__ == "__main__":
    with_text = True
    df = pd.read_csv("data/reduced_m5_dataset.csv")
    df = downcast(df)
    df = preprocess_data(df, with_text)

    num_leaves_list = [31, 63, 127, 255, 511]
    colsample_bytree_list = [0.6, 1]

    all_params = []
    best_param = None
    best_score = 9999
    for num_leaves in num_leaves_list:
        for colsample_bytree in colsample_bytree_list:
            root_mse, mae, val_rmse = train(df, num_leaves=num_leaves, colsample_bytree=colsample_bytree)
            all_params.append(
                {
                    "num_leaves": num_leaves,
                    "colsample_bytree": colsample_bytree,
                    "mae": mae,
                    "rmse": root_mse,
                    "val_rmse": val_rmse,
                }
            )
            if val_rmse < best_score:
                best_score = val_rmse
                best_param = all_params[-1]

    json.dump(all_params, open(f"lgbm_{with_text}.json", "w"))
    print(best_param)
