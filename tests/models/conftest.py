"""Setup for model tests."""

import pytest

parametrize_dtype = pytest.mark.parametrize("dtype", [None, "float32", "float64"])
