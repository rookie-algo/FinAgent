#!/usr/bin/env python
"""
Bootstrap script for Django SaaS project.
Runs migrations, creates admin, starts server.
"""
import logging
import os
import sys
import subprocess

from pathlib import Path
from dotenv import load_dotenv


# Load .env
load_dotenv()

# Add project to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")

import django
django.setup()

from django.contrib.auth import get_user_model
from django.db import connection

User = get_user_model()


def run_command(cmd, check=True):
    logging.info(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if check and result.returncode != 0:
        sys.exit(result.returncode)


def create_superuser():
    """Create admin user if not exists"""
    username = os.getenv("DJANGO_SUPERUSER_NAME")
    admin_email = os.getenv("DJANGO_SUPERUSER_EMAIL")
    admin_password = os.getenv("DJANGO_SUPERUSER_PASSWORD")

    if not admin_email or not admin_password:
        logging.info("Skipping superuser creation (DJANGO_SUPERUSER_EMAIL/PASSWORD not set)")
        return

    if not User.objects.filter(email=admin_email).exists():
        logging.info(f"Creating superuser: {admin_email}")
        User.objects.create_superuser(
            username=username,
            email=admin_email,
            password=admin_password
        )
    else:
        logging.info(f"Superuser {admin_email} already exists")


def run_migrations():
    """Run makemigrations + migrate"""
    logging.info("Applying migrations...")
    run_command(["uv", "run", "python", "manage.py", "makemigrations"])
    run_command(["uv", "run", "python", "manage.py", "migrate"])


def main():
    # 1. Run migrations
    run_migrations()

    # 2. Create superuser
    create_superuser()

    # 3. Start server
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        logging.info("Starting Gunicorn (production mode)...")
        cmd = [
            "uv", "run", "gunicorn",
            "app.wsgi:app",
            "--bind", "0.0.0.0:8000",
            "--workers", "4",
            "--log-level", "info"
        ]
    else:
        logging.info("Starting Django dev server...")
        cmd = ["uv", "run", "python", "manage.py", "runserver", "0.0.0.0:8000"]

    # Execute server
    run_command(cmd, check=False)  # Let server run


if __name__ == "__main__":
    main()