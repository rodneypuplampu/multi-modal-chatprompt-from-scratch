# settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Google Cloud Storage settings
GS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
GS_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
DEFAULT_FILE_STORAGE = 'storages.backends.gcloud.GoogleCloudStorage'

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'channels',
    'chat',
]
