rm db.sqlite3
mv accounts/migrations/0001_create_superuser.py .
rm **/migrations/0*
python manage.py makemigrations
mv 0001_create_superuser.py accounts/migrations/0001_create_superuser.py
python manage.py migrate
# python manage.py loaddata cars.json
