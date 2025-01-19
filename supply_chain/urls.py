"""supply_chain URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
import forecasting.views as forecasting_views
import authentication.views as authentication_views

urlpatterns = [
    path("signup/", authentication_views.signup_view, name="signup"),
    path("login/", authentication_views.login_view, name="login"),
    path("logout/", authentication_views.logout_view, name="logout"),
    path("register_company/", forecasting_views.register_company, name="register_company"),
    path("upload_sales/", forecasting_views.upload_sales, name="upload_sales"),
    path("locate_stores/", forecasting_views.locate_stores, name="locate_stores"),
    path("", forecasting_views.dashboard, name="dashboard"),
    path("run_forecast/", forecasting_views.run_forecast, name="run_forecast"),

]