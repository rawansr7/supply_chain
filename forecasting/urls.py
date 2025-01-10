from django.urls import path
from . import views

urlpatterns = [
    path("register_company/", views.register_company, name="register_company"),
    path("upload_sales/", views.upload_sales, name="upload_sales"),
    path("locate_stores/", views.locate_stores, name="locate_stores"),
    path("", views.dashboard, name="dashboard"),
    path("run_forecast/", views.run_forecast, name="run_forecast"),
]
