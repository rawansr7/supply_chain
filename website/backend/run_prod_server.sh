python manage.py migrate
python manage.py collectstatic --verbosity 2 --no-input
gunicorn --bind 0.0.0.0:8000 rswebsite.wsgi:application
