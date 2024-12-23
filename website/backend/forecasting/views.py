from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .training import forecast_next_week
from .utils import export_company_data, save_forecasting_results


@login_required
def run_forecast(request):
    company = request.user.company
    data = export_company_data(company)
    forecast_results = forecast_next_week(data)
    save_forecasting_results(forecast_results)
    return JsonResponse(forecast_results, safe=False)
