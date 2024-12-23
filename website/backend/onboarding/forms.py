from django import forms
from .models import Product, Sale, Company


class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ["name", "sku"]


class SaleForm(forms.ModelForm):
    class Meta:
        model = Sale
        fields = ["store", "product", "quantity", "sale_date"]


class CompanyForm(forms.ModelForm):
    class Meta:
        model = Company
        fields = ["name", "address"]
