#!/usr/bin/env python
"""
Quick setup script to create the database if it doesn't exist.
Run this after adding database configuration to settings.py
"""

import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'scchatbot.settings')

try:
    import django
    django.setup()
    
    from django.core.management import execute_from_command_line
    
    print("Creating database tables...")
    execute_from_command_line(['manage.py', 'migrate'])
    print("✅ Database setup complete!")
    
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nPlease make sure you're in the correct conda environment:")
    print("conda activate ScChatbot")
    sys.exit(1)