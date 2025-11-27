"""Setup for impulse model tests."""

from abc import ABC
from abc import abstractmethod
import pandas as pd
import pytest
from prfmodel.models.base import BaseImpulse
from tests.models.conftest import parametrize_dtype


class TestImpulseSetup(ABC):
    """Parameters for impulse response model tests."""

    duration = 32
    offset = 0.0001
    resolution = 1.0

    @pytest.fixture
    @abstractmethod
    def irf_model(self):
        """Impulse response model object."""

    def test_num_frames(self, irf_model: BaseImpulse):
        """Test that property num_frames is correct."""
        assert irf_model.num_frames == int(self.duration / self.resolution)

    def test_frames(self, irf_model: BaseImpulse):
        """Test that property frames has correct shape."""
        assert irf_model.frames.shape == (1, irf_model.num_frames)

    @parametrize_dtype
    def test_call(self, irf_model: BaseImpulse, parameters: pd.DataFrame, dtype: str):
        """Test that model response has correct shape."""
        resp = irf_model(parameters, dtype)

        assert resp.shape == (parameters.shape[0], irf_model.frames.shape[1])
