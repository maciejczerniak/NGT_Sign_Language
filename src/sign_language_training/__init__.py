"""Standalone training package for the NGT sign-language model.

This package is intentionally decoupled from ``sign_language`` so the
training Azure ML job can install a minimal environment without any
inference-API dependencies (uvicorn, mediapipe, websockets, fastapi, ...).
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
