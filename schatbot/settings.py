# settings.py

INSTALLED_APPS = [
    # ...
    'django.contrib.sessions',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'schatbot',  # Make sure to replace with your actual app name
    # ...
]

MIDDLEWARE = [
    # ...
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # ...
]

# Use database-backed sessions
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
