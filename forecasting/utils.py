from .models import Sale, Store, Product, ForecastResult


def export_company_data(company):
    sales = Sale.objects.filter(store__company=company)
    data = []
    for sale in sales:
        data.append(
            {
                "store_id": sale.store.name,
                "item_id": sale.product.name,
                "sold": sale.sold,
                "date": sale.date,
            }
        )
    return data


def save_forecasting_results(forecast_results, company):
    for result in forecast_results:
        store = result["store_id"]
        product = result["product_id"]
        forecasted_sold = result["forecasted_sold"]
        forecasted_date = result["forecasted_date"]

        store_obj = Store.objects.get(name=store, company=company)
        product_obj = Product.objects.get(name=product, company=company)
        forecast_result = ForecastResult(
            store=store_obj,
            product=product_obj,
            forecasted_sold=forecasted_sold,
            forecasted_date=forecasted_date,
        )
        forecast_result.save()


def aggregate_forecasts(forecast_results):
    aggregated_results = {}
    for result in forecast_results:
        store = result["store_id"]
        product = result["product_id"]
        forecasted_sold = result["forecasted_sold"]
        if store not in aggregated_results:
            aggregated_results[store] = {}
        if product not in aggregated_results[store]:
            aggregated_results[store][product] = 0
        aggregated_results[store][product] += forecasted_sold

    return aggregated_results
