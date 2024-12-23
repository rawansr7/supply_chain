from django.urls import path
from . import views

urlpatterns = [
    path("register_company/", views.register_company, name="register_company"),
    path("add_stores/", views.add_stores, name="add_stores"),
    path("add_products/", views.add_products, name="add_products"),
    path("upload_sales/", views.upload_sales, name="upload_sales"),
]
