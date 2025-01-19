import pandas as pd
import numpy as np

np.random.seed(30061998)


num_products = 100
num_stores = 4

sales = pd.read_csv("../data/sales_train_evaluation.csv")
calendar = pd.read_csv("../data/calendar.csv")

products = sales["item_id"].unique().tolist()
stores = sales["store_id"].unique().tolist()

chosen_item_ids = np.random.choice(products, size=num_products, replace=False)
chosen_store_ids = np.random.choice(stores, size=num_stores, replace=False)

sales = sales[sales["item_id"].isin(chosen_item_ids)]
sales = sales[sales["store_id"].isin(chosen_store_ids)]

sales = sales.drop(columns=["id", "dept_id", "state_id", "cat_id"])

sales = pd.melt(sales, id_vars=["item_id", "store_id"], var_name="d", value_name="sold").dropna()


calendar = calendar[["d", "date"]]
sales = pd.merge(sales, calendar, how="left", on="d")
sales = sales.drop(columns=["d"])

sales.to_csv("../data/reduced_m5_dataset.csv", index=False)
