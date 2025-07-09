# settings.py
SECRET_KEY = 'vGC7Ma3yatftileSw4kw44aCdL04FSUtoqoKux_X4ws0TtLumScxvI8QAjWwLMWkJK8'
INSTALLED_APPS = [
    # ...
    'django.contrib.sessions',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'scchatbot',  # Make sure to replace with your actual app name
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

DEBUG = True

from pathlib import Path
ROOT_URLCONF = "scchatbot.urls"
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_URL = "/static/"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  # or another path if needed
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]