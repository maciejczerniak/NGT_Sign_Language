import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Sign Language"
copyright = "2026, Kinga Marchlewska, Raya Kichukova, Ana-Maria Farazică, Maciej Czerniak, Szymon Chirowski"
author = "Kinga Marchlewska, Raya Kichukova, Ana-Maria Farazică, Maciej Czerniak, Szymon Chirowski"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_mock_imports = [
    "sqlalchemy",
    "fastapi_users",
    "fastapi_users_db_sqlalchemy",
    "azure",
    "azure.ai",
    "azure.ai.ml",
    "azure.identity",
    "mlflow",
    "torch",
    "torchvision",
    "mediapipe",
    "cv2",
    "numpy",
    "sklearn",
    "PIL",
    "websockets",
    "pydantic",
    "pydantic_settings",
    "typer",
    "uvicorn",
    "fastapi",
    "alembic",
    "matplotlib",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/tests/**",
]

html_theme = "sphinx_rtd_theme"
html_static_path = []


def _is_pydantic_validator_warning(message: str) -> bool:
    """Return True for the known Pydantic v2 field_validator signature warning."""
    return "Failed to get a method signature for" in message and (
        "is not a callable object" in message
    )


def setup(app):
    """Connect a targeted warning filter for known Pydantic v2 incompatibilities."""
    import sphinx.util.logging
    import logging

    class PydanticValidatorFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not _is_pydantic_validator_warning(record.getMessage())

    sphinx_logger = sphinx.util.logging.getLogger("sphinx.ext.autodoc")
    sphinx_logger.logger.addFilter(PydanticValidatorFilter())
