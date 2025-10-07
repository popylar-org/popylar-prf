"""Tests for encoder classes and methods."""

import numpy as np
import pytest
from prfmodel.models.encoding import encode_prf_response


class TestEncoder:
    """Tests for the Encoder class."""

    num_voxels: int = 3
    num_frames: int = 10

    width: int = 5
    height: int = 4
    depth: int = 3

    def _check_encoding(self, x: np.ndarray) -> None:
        assert x.shape == (self.num_voxels, self.num_frames)
        # Check that all voxels have identical encoding for identical response and design
        assert np.unique(x, axis=1).shape == (self.num_voxels, 1)

    @pytest.fixture
    def response_1d(self):
        """1D model response."""
        resp_dummy = np.ones((self.num_voxels, self.height))
        # Response is function of height
        return resp_dummy * np.expand_dims(np.sin(np.arange(self.height)), 0)

    @pytest.fixture
    def response_2d(self):
        """2D model response."""
        resp_dummy = np.ones((self.num_voxels, self.height, self.width))
        # Response is function of width
        return resp_dummy * np.expand_dims(np.sin(np.arange(self.width)), (0, 1))

    @pytest.fixture
    def response_3d(self):
        """3D model response."""
        resp_dummy = np.ones((self.num_voxels, self.height, self.width, self.depth))
        # Response is function of depth
        return resp_dummy * np.expand_dims(np.sin(np.arange(self.depth)), (0, 1, 2))

    def test_call_1d(self, response_1d: np.ndarray):
        """Test that 1D encoding returns the correct shape."""
        design = np.ones((self.num_frames, self.height))

        x = np.asarray(encode_prf_response(response_1d, design))

        self._check_encoding(x)

    def test_call_2d(self, response_2d: np.ndarray):
        """Test that 2D encoding returns the correct shape."""
        design = np.ones((self.num_frames, self.height, self.width))

        x = np.asarray(encode_prf_response(response_2d, design))

        self._check_encoding(x)

    def test_call_3d(self, response_3d: np.ndarray):
        """Test that 3D encoding returns the correct shape."""
        design = np.ones((self.num_frames, self.height, self.width, self.depth))

        x = np.asarray(encode_prf_response(response_3d, design))

        self._check_encoding(x)
