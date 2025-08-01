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

# Disable caching in development
if DEBUG:
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        }
    }
    
    # Add custom middleware to disable browser caching for static files
    MIDDLEWARE.append('scchatbot.middleware.DisableBrowserCacheMiddleware')
    
    # Cache settings
    CACHE_MIDDLEWARE_SECONDS = 0  # Don't cache anything
    CACHE_MIDDLEWARE_KEY_PREFIX = ''

from pathlib import Path
ROOT_URLCONF = "scchatbot.urls"
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_URL = "/static/"

# For production: Use ManifestStaticFilesStorage for automatic cache busting
# STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'
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