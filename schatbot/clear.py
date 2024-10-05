# clear.py

import os
import django
from django.conf import settings
from django.contrib.sessions.models import Session

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'newui.settings')
django.setup()

def clear_current_session():
    try:
        # Fetch all sessions
        sessions = Session.objects.all()

        # Clear all session data
        sessions.delete()
        print("Successfully cleared all sessions.")
    except Exception as e:
        print(f"Error clearing sessions: {e}")

if __name__ == "__main__":
    clear_current_session()
