import pandas as pd

# Load the sales dataset
sales_file = "../data/sales_train_evaluation.csv"  # Update with the path to your file
sales_data = pd.read_csv(sales_file)

# Extract unique products with their department and category IDs
# Assuming dept_id and cat_id are in the file with item_id
unique_products = sales_data[["item_id", "dept_id", "cat_id"]].drop_duplicates()

# Save the unique products to a new CSV file
output_file = "../data/unique_products.csv"
unique_products.to_csv(output_file, index=False)
