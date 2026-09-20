from django.conf import settings


def google_maps_api_key(request):
    """Make the Maps key available to the templates that load the Maps JS API."""
    return {"GOOGLE_MAPS_API_KEY": settings.GOOGLE_MAPS_API_KEY}
