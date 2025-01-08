from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from analysis.forecast import forecast_next_month
from .utils import export_company_data, save_forecasting_results
from django.shortcuts import render, redirect
from .models import Store, Product, Sale, Company
import csv


@login_required
def register_company(request):
    if request.method == "POST":
        name = request.POST.get("name")
        address = request.POST.get("address")
        company = Company(user=request.user, name=name, address=address)
        company.save()
        return redirect("add_stores")

    if request.method == "GET":
        return render(request, "register_company.html")


@login_required
def add_stores(request):
    if request.method == "POST":
        name = request.POST.get("name")
        geolocation = request.POST.get("geolocation")
        Store.objects.create(company=request.user.company, name=name, geolocation=geolocation)
    stores = Store.objects.filter(company=request.user.company)
    return render(request, "add_stores.html", {"stores": stores})


@login_required
def add_products(request):
    company = request.user.company

    if request.method == "POST":
        csv_file = request.FILES.get("file")
        decoded_file = csv_file.read().decode("utf-8").splitlines()
        reader = csv.DictReader(decoded_file)
        for row in reader:
            Product.objects.create(company=request.user.company, name=row["name"])
        return redirect("upload_sales")

    if request.method == "GET":
        products = Product.objects.filter(company=company)
        return render(request, "add_products.html", {"products": products})


@login_required
def upload_sales(request):
    if request.method == "POST":
        csv_file = request.FILES.get("file")
        decoded_file = csv_file.read().decode("utf-8").splitlines()
        reader = csv.DictReader(decoded_file)
        for row in reader:
            store_object = Store.objects.get(name=row["store"], company=request.user.company)
            product_object = Product.objects.get(name=row["product"], company=request.user.company)
            Sale.objects.create(
                store=store_object,
                product=product_object,
                quantity=row["quantity"],
                sale_date=row["sale_date"],
            )
        return redirect("dashboard")

    if request.method == "GET":
        sales = Sale.objects.filter(store__company=request.user.company)
        return render(
            request,
            "upload_sales.html",
            {"sales": sales},
        )


@login_required
def run_forecast(request):
    company = request.user.company
    data = export_company_data(company)
    forecast_results = forecast_next_month(data)
    save_forecasting_results(forecast_results)
    return JsonResponse(forecast_results, safe=False)


@login_required
def dashboard(request):
    try:
        company = Company.objects.get(user__id=request.user.id)
    except Company.DoesNotExist:
        return redirect("register_company")

    stores = Store.objects.filter(company=company)
    products = Product.objects.filter(company=company)

    # Prepare store data for Google Maps
    stores_data = [
        {
            "name": store.name,
            "lat": store.geolocation.lat,
            "lng": store.geolocation.lon,
        }
        for store in stores
    ]

    return render(
        request,
        "dashboard.html",
        {
            "company": company,
            "products": products,
            "stores": stores_data,
        },
    )
