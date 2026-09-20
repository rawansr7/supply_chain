: "${GOOGLE_MAPS_API_KEY:?not set - the maps will not load. e.g. export GOOGLE_MAPS_API_KEY=\$(cat ../maps.key)}"
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --no-input
python manage.py runserver
