rm db.sqlite3
rm **/migrations/0*
python manage.py makemigrations
python manage.py migrate
