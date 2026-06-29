"""FastAPI microservice for triggering Azure ML retraining jobs."""

from sign_language_training.trigger_api.app import create_app

__all__ = ["create_app"]
