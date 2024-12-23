from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from onboarding.models import Store, Product, Company


def home(request):
    return render(request, "home.html")


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
