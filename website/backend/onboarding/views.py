from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Store, Product, Sale, Company
from django.core.exceptions import BadRequest
from .forms import ProductForm, SaleForm, CompanyForm
import csv


@login_required
def register_company(request):
    if request.method == "POST":
        form = CompanyForm(request.POST)
        if form.is_valid():
            company = form.save(commit=False)
            company.user = request.user
            company.save()
            return redirect("add_stores")
        raise BadRequest()
    try:
        Company.objects.get(user__id=request.user.id)
        return redirect("add_stores")
    except Company.DoesNotExist:
        pass
    form = CompanyForm()
    return render(request, "register_company.html", {"form": form})


@login_required
def add_stores(request):
    if request.method == "POST":
        name = request.POST.get("name")
        address = request.POST.get("address")
        geolocation = request.POST.get("geolocation")
        Store.objects.create(
            company=request.user.company,
            name=name,
            address=address,
            geolocation=geolocation,
        )
    stores = Store.objects.filter(company=request.user.company)
    return render(request, "add_stores.html", {"stores": stores})


@login_required
def add_products(request):
    company = request.user.company

    if request.method == "POST":
        # Handle single product addition
        if "add_single_product" in request.POST:
            form = ProductForm(request.POST)
            if form.is_valid():
                product = form.save(commit=False)
                product.company = company
                product.save()

        # Handle CSV upload
        elif "upload_csv" in request.POST:
            csv_file = request.FILES.get("file")
            if csv_file:
                decoded_file = csv_file.read().decode("utf-8").splitlines()
                reader = csv.DictReader(decoded_file)
                for row in reader:
                    form = ProductForm(row)
                    if form.is_valid():
                        product = form.save(commit=False)
                        product.company = company
                        if not Product.objects.filter(
                            name=product.name, company=company
                        ).exists():
                            product.save()

    products = Product.objects.filter(company=company)
    product_form = ProductForm()
    return render(
        request,
        "add_products.html",
        {"products": products, "product_form": product_form},
    )


@login_required
def upload_sales(request):
    company = request.user.company

    if request.method == "POST":
        # Handle single sale addition
        if "add_single_sale" in request.POST:
            form = SaleForm(request.POST)
            if form.is_valid():
                sale = form.save(commit=False)
                if sale.store.company == company and sale.product.company == company:
                    sale.save()

        # Handle CSV upload
        elif "upload_csv" in request.POST:
            csv_file = request.FILES.get("file")
            if csv_file:
                decoded_file = csv_file.read().decode("utf-8").splitlines()
                reader = csv.DictReader(decoded_file)
                for row in reader:
                    form = SaleForm(row)
                    if form.is_valid():
                        sale = form.save(commit=False)
                        if (
                            sale.store.company == company
                            and sale.product.company == company
                        ):
                            sale.save()

    sales = Sale.objects.filter(store__company=company).select_related(
        "store", "product"
    )
    stores = Store.objects.filter(company=company)
    products = Product.objects.filter(company=company)
    sale_form = SaleForm()
    return render(
        request,
        "upload_sales.html",
        {
            "sales": sales,
            "stores": stores,
            "products": products,
            "sale_form": sale_form,
        },
    )
