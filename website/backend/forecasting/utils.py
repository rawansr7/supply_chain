from onboarding.models import Sale, Store, Product
from .models import ForecastResult


def export_company_data(company):
    sales = Sale.objects.filter(store__company=company)
    data = []
    for sale in sales:
        data.append(
            {
                "store": sale.store.name,
                "product": sale.product.name,
                "quantity": sale.quantity,
                "sale_date": sale.sale_date,
            }
        )
    return data


def save_forecasting_results(forecast_results):
    company = forecast_results[0]["store"].company
    for result in forecast_results:
        store = result["store"]
        product = result["product"]
        forecasted_quantity = result["forecasted_quantity"]

        store_obj = Store.objects.get(name=store, company=company)
        product_obj = Product.objects.get(name=product, company=company)
        forecast_result = ForecastResult(
            store=store_obj,
            product=product_obj,
            forecasted_quantity=forecasted_quantity,
        )
        forecast_result.save()
