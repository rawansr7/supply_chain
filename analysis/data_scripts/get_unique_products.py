import pandas as pd

# Load the sales dataset
sales_data = pd.read_csv("../data/sales_train_evaluation.csv")

# Extract unique products with their department and category IDs
# Assuming dept_id and cat_id are in the file with item_id
unique_products = sales_data[["item_id", "dept_id", "cat_id"]].drop_duplicates()

# Save the unique products to a new CSV file
unique_products.to_csv("../data/unique_products.csv", index=False)
