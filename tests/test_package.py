"""Smoke tests — can we import the package at all?"""


def test_import_package():
    """Package should be importable without errors."""
    import sign_language

    assert hasattr(sign_language, "__version__")


def test_version_is_string():
    """Version should be a string."""
    from sign_language import __version__

    assert isinstance(__version__, str)
