from django.apps import AppConfig


class InferenceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fraud_detection_web.inference'
