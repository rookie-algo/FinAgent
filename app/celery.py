import os
from celery import Celery


# Set default Django settings module for Celery
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")

# Initialize Celery application instance.
app = Celery("app")

# Load the configuration from django.settings file
app.config_from_object("django.conf:settings", namespace="CELERY")

# Automatically discover task modules in all registered Django apps.
app.autodiscover_tasks()
