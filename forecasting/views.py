from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .utils import export_company_data, save_forecasting_results, aggregate_forecasts
from django.shortcuts import render, redirect
from .models import Store, Product, Sale, Company
import pandas as pd


@login_required
def register_company(request):
    if request.method == "POST":
        name = request.POST.get("name")
        company = Company(user=request.user, name=name)
        company.save()
        return redirect("upload_sales")

    if request.method == "GET":
        return render(request, "register_company.html")


@login_required
def upload_sales(request):
    if request.method == "POST":
        csv_file = request.FILES.get("file")
        data = pd.read_csv(csv_file)
        stores_objects = {}
        products_objects = {}
        for product_id in data["item_id"].unique():
            product_object = Product.objects.create(name=product_id, company=request.user.company)
            products_objects[product_id] = product_object
        for store_id in data["store_id"].unique():
            store_object = Store.objects.create(name=store_id, company=request.user.company)
            stores_objects[store_id] = store_object

        sales_to_create = data.apply(
            lambda row: Sale(
                store=stores_objects[row["store_id"]],
                product=products_objects[row["item_id"]],
                sold=row["sold"],
                date=row["date"],
            ),
            axis="columns",
        ).tolist()

        Sale.objects.bulk_create(sales_to_create)

        return redirect("dashboard")

    if request.method == "GET":
        sales = Sale.objects.filter(store__company=request.user.company)
        return render(
            request,
            "upload_sales.html",
            {"sales": sales},
        )


@login_required
def locate_stores(request):
    if request.method == "POST":
        for store_id, geolocation in request.POST.items():
            if store_id == "csrfmiddlewaretoken":
                continue
            store = Store.objects.get(name=store_id, company=request.user.company)
            store.geolocation = geolocation
            store.save()

        return redirect("dashboard")

    # Pass stores with initial locations to the template
    stores = Store.objects.filter(company=request.user.company)
    stores = [
        {
            "name": store.name,
            "geolocation": {
                "lat": store.geolocation.lat,
                "lon": store.geolocation.lon,
            },
        }
        for store in stores
    ]

    return render(
        request,
        "locate_stores.html",
        {
            "stores": stores,
        },
    )


@login_required
def dashboard(request):
    try:
        company = Company.objects.get(user__id=request.user.id)
    except Company.DoesNotExist:
        return redirect("register_company")

    products = Product.objects.filter(company=company)
    stores = Store.objects.filter(company=company)
    stores = [
        {
            "name": store.name,
            "geolocation": {
                "lat": store.geolocation.lat,
                "lon": store.geolocation.lon,
            },
        }
        for store in stores
    ]
    return render(
        request,
        "dashboard.html",
        {
            "company": company,
            "products": products,
            "stores": stores,
        },
    )


@login_required
def run_forecast(request):
    from analysis.forecast import forecast_next_month

    company = request.user.company
    data = export_company_data(company)
    forecast_results = forecast_next_month(data)
    save_forecasting_results(forecast_results, company)
    forecast_results = aggregate_forecasts(forecast_results)
    return JsonResponse(forecast_results, safe=False)
