"""
SMOKE TESTS
Minimal checks for package import and Python version.
"""


def test_riskml_import():
    import riskml
    assert riskml is not None


def test_python_version():
    import sys
    assert sys.version_info >= (3, 11)
