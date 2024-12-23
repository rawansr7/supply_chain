from django.db import models
from django.contrib.auth.models import User
from django_google_maps.fields import AddressField, GeoLocationField


class Company(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    address = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Store(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    address = AddressField(max_length=200)
    geolocation = GeoLocationField()

    def __str__(self):
        return self.name


class Product(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    sku = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class Sale(models.Model):
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    sale_date = models.DateField()

    def __str__(self):
        return f"{self.store} - {self.product} - {self.quantity}"
