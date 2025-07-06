# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Tests for the sdk package's __init__.py"""


def test_sdk_import():
    """Test that the sdk package can be imported."""
    import sdk

    assert sdk is not None

    # Verify expected attributes are available
    assert hasattr(sdk, "engine")
    assert hasattr(sdk, "rest_api")

    # Verify the package has a version
    assert hasattr(sdk, "__version__")
    assert isinstance(sdk.__version__, str)
    assert len(sdk.__version__) > 0
