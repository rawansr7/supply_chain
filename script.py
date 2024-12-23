import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Load the dataset (after downloading manually or via Kaggle API)
sales_data = pd.read_csv("sales_train_evaluation.csv")
calendar_data = pd.read_csv("calendar.csv")
prices_data = pd.read_csv("sell_prices.csv")

# Inspect the dataset
print(sales_data.head())
print("bbb\n\n")
# Melt sales data to long format for time-series processing
sales_data_long = sales_data.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d",
    value_name="sales",
)
print(sales_data_long.head())
input("hh")


# Merge with calendar data to get actual dates
sales_data_long = sales_data_long.merge(
    calendar_data, how="left", left_on="d", right_on="d"
)

# Merge with price data for additional features
sales_data_long = sales_data_long.merge(
    prices_data, how="left", on=["store_id", "item_id", "wm_yr_wk"]
)

# Drop unnecessary columns and handle missing data
sales_data_long = sales_data_long.drop(
    columns=[
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_name_2",
        "event_type_1",
        "event_type_2",
    ]
)
sales_data_long.fillna(0, inplace=True)

# Add lag features (e.g., previous sales)
sales_data_long["lag_7"] = sales_data_long.groupby("id")["sales"].shift(7)
sales_data_long["lag_14"] = sales_data_long.groupby("id")["sales"].shift(14)

# Add rolling mean features
sales_data_long["rolling_mean_7"] = (
    sales_data_long.groupby("id")["sales"].shift(7).rolling(7).mean()
)

# Drop rows with NaN values after feature engineering
sales_data_long.dropna(inplace=True)

print(sales_data_long.head())

#########################################################

# Select features and target
features = [
    "sales",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]
target = "sales"

# Train-test split
X = sales_data_long[features]
y = sales_data_long[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=6
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate models
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
