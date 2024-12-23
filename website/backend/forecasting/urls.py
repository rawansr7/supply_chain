from django.urls import path
from . import views

urlpatterns = [
    path("run_forecast/", views.run_forecast, name="run_forecast"),
]
