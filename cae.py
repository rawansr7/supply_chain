from pycaret.time_series import *
import pandas as pd

# Initialize PyCaret
sales_data = pd.read_csv("sales_train_evaluation.csv")
sales_data_long = sales_data.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d",
    value_name="sales",
)
reg_setup = setup(data=sales_data_long, target='sales')

# Compare and find the best model
best_model = compare_models()

# Finalize the best model
final_model = finalize_model(best_model)

# Predict on new data
predictions = predict_model(final_model)