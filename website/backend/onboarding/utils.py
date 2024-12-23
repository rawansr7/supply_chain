import pandas as pd
from pycaret.regression import *
import pandas as pd

def preprocess_inventory_data(file_path):
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df.drop(columns=['snap_CA', 'snap_TX', 'snap_WI'], errors='ignore', inplace=True)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Generate temporal features
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    return df


# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Define PyCaret experiment
experiment = setup(data=data, target='sales', session_id=123)

# Compare models and select the best
best_model = compare_models()

# Save the best model
save_model(best_model, 'inventory_forecast_model')
