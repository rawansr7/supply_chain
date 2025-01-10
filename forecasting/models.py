from django.db import models
from django.contrib.auth.models import User
from django_google_maps.fields import GeoLocationField


class Company(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)


class Store(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    geolocation = GeoLocationField(default="33.8938,35.5018")


class Product(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)


class Sale(models.Model):
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    sold = models.PositiveIntegerField()
    date = models.DateField()


class ForecastResult(models.Model):
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    forecasted_sold = models.PositiveIntegerField()
    forecasted_date = models.DateField()
    forecasted_at = models.DateField(auto_now_add=True)
