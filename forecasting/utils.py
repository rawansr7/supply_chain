from .models import Sale, Store, Product, ForecastResult
import pandas as pd


def export_company_data(company):
    sales = Sale.objects.filter(store__company=company).values("store__name", "product__name", "sold", "date")
    df = pd.DataFrame.from_records(sales)
    df.rename(columns={"store__name": "store_id", "product__name": "item_id"}, inplace=True)
    return df


def save_forecasting_results(forecasted_data, company):
    stores_objects = {}
    products_objects = {}
    for product_id in forecasted_data["item_id"].unique():
        product_object = Product.objects.get(name=product_id, company=company)
        products_objects[product_id] = product_object
    for store_id in forecasted_data["store_id"].unique():
        store_object = Store.objects.get(name=store_id, company=company)
        stores_objects[store_id] = store_object

    results_to_create = forecasted_data.apply(
        lambda row: ForecastResult(
            store=stores_objects[row["store_id"]],
            product=products_objects[row["item_id"]],
            forecasted_sold=row["forecasted_sold"],
            forecasted_date=row["forecasted_date"],
        ),
        axis="columns",
    ).tolist()

    ForecastResult.objects.bulk_create(results_to_create)


def aggregate_forecasts(forecasted_data):
    forecasted_data.drop(columns=["forecasted_date"], inplace=True)
    forecasted_data = forecasted_data.groupby(["item_id", "store_id"], as_index=False).agg({"forecasted_sold": "sum"})
    forecasted_data = forecasted_data.query("forecasted_sold != 0")

    data_as_dict = {}
    for index, row in forecasted_data.iterrows():
        store = row["store_id"]
        product = row["item_id"]
        forecasted_sold = row["forecasted_sold"]
        if store not in data_as_dict:
            data_as_dict[store] = {}
        data_as_dict[store][product] = forecasted_sold

    return data_as_dict
