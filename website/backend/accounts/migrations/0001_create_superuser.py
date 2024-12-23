from django.db import migrations
from django.conf import settings
from django.contrib.auth.models import User

def createsuperuser(apps, schema_editor) -> None:
    """
    Dynamically create an admin user as part of a migration
    """

    User.objects.create_superuser(
        settings.SUPERUSER_USERNAME, password=settings.SUPERUSER_PASSWORD
    )


class Migration(migrations.Migration):

    initial = True
    dependencies = [
        ("auth", "0012_alter_user_first_name_max_length"),
    ]
    operations = [migrations.RunPython(createsuperuser)]
