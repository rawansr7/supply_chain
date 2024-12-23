from django.db import models
from onboarding.models import Store, Product


class ForecastResult(models.Model):
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    forecasted_quantity = models.PositiveIntegerField()
    forecast_date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.store.name} - {self.product.name} - {self.forecasted_quantity}"
