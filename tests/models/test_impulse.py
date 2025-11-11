"""Tests for impulse response model classes and functions."""

from itertools import product
import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy import special
from prfmodel.models.impulse import TwoGammaImpulse
from prfmodel.models.impulse import convolve_prf_impulse_response
from prfmodel.models.impulse import gamma_density
from .conftest import parametrize_dtype


def test_convolve_prf_impulse_response():
    """Test that convolve_prf_impulse_response returns response with correct shape."""
    num_batches = 3
    num_prf_frames = 10
    num_irf_frames = 3

    prf_response = np.ones((num_batches, num_prf_frames))
    irf_response = np.ones((num_batches, num_irf_frames))

    resp_conv = convolve_prf_impulse_response(prf_response, irf_response)

    assert resp_conv.shape == (num_batches, num_prf_frames)


class TestIRFSetup:
    """Setup parameters for impulse response model testing."""

    duration = 32
    offset = 0.0001
    resolution = 0.1


class TestGammaDensity(TestIRFSetup):
    """Tests for calc_gamma_density function."""

    @pytest.fixture
    def frames(self):
        """Time frames."""
        # Frames must have shape (n, 1)
        return np.expand_dims(np.linspace(self.offset, self.duration, int(self.duration / self.resolution)), 0)

    @pytest.fixture
    def parameter_range(self):
        """Range of shape and rate parameters."""
        return np.round(np.linspace(0.1, 5.0, 5), 2)

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray):
        """Shape and rate parameter combinations."""
        return np.array(list(product(parameter_range, parameter_range)))

    def test_gamma_density(self, frames: np.ndarray, parameters: np.ndarray):
        """
        Test that gamma density peaks at correct frame across combinations of shape and rate parameters.

        Argument `parameters` is a two-dimensional array where the first column is the shape and the second column
        the rate parameter of each parameter combination.

        The peak of the gamma density is tested against the expected analytical mode of the gamma distribution.

        """
        # Parameters must have shape (n, 1)
        shape = np.expand_dims(parameters[:, 0], 1)
        rate = np.expand_dims(parameters[:, 1], 1)

        resp = np.asarray(gamma_density(frames, shape, rate))

        assert np.all(resp > 0.0)

        frames = frames.squeeze()
        # Calc expected analytical mode of each parameter combination
        expected_mode = np.where(shape < 1, 0, (shape - 1) / rate).squeeze()
        # Find the frame of the peak response
        peak_frame_idx = np.argmax(resp, axis=1)
        # Find the peak response of each parameter combination
        peak_response = frames[peak_frame_idx]

        for ep, pr, idx in zip(expected_mode, peak_response, peak_frame_idx, strict=False):
            # If the expected mode is beyond the last frame, the peak response should be in the last frame
            if ep >= frames.max():
                assert idx == (len(frames) - 1), "Peak response must be in last frame"
            # If the expected mode is before the first frame, the peak response should be in the first frame
            elif ep <= frames.min():
                assert idx == 0, "Peak response must be in first frame"
            # Difference between expected mode and peak response should not be larger than frame resolution
            else:
                assert abs(pr - ep) <= self.resolution

    def test_gamma_density_integral(self):
        """Test that the integral of normalized density is 1."""
        integ = integrate.quad(gamma_density, 0, np.inf, args=(2.0, 1.0, True))

        assert integ[0] == pytest.approx(1.0)

    def test_gamma_density_unnormalized(self, frames: np.ndarray):
        """Test that the normalized density is equal to the unnormalized density times the normalizing constant."""
        shape = np.array([[2.0]])
        rate = np.array([[1.0]])
        dens_norm = np.asarray(gamma_density(frames, shape, rate))
        dens_unnorm = np.asarray(gamma_density(frames, shape, rate, norm=False))

        assert np.all(dens_norm == dens_unnorm * (rate**shape / special.gamma(shape)))

    def test_values_value_error(self, frames: np.ndarray):
        """Test that negative values raise an error."""
        shape = np.array([[1.0, 2.0]])
        rate = np.array([[1.0, 1.0]])
        frames[0] = -1.0

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_shape_value_error(self, frames: np.ndarray):
        """Test that negative shape parameters raise an error."""
        shape = np.array([[-1.0, 2.0]])
        rate = np.array([[1.0, 1.0]])

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_rate_value_error(self, frames: np.ndarray):
        """Test that negative rate parameters raise an error."""
        shape = np.array([[1.0, 2.0]])
        rate = np.array([[-1.0, 1.0]])

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)


class TestTwoGammaResponseModel(TestIRFSetup):
    """Tests for TwoGammaResponseModel class."""

    @pytest.fixture
    def parameter_range(self):
        """Range of parameters."""
        return np.round(np.linspace(0.1, 5.0, 3), 2)

    @pytest.fixture
    def weight_range(self):
        """Range of weigth parameter."""
        return np.linspace(0.0, 1.0, 3)

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray, weight_range: np.ndarray):
        """Model parameter combinations."""
        values = np.array(
            list(
                product(
                    parameter_range,
                    parameter_range,
                    parameter_range,
                    parameter_range,
                    weight_range,
                ),
            ),
        )
        return pd.DataFrame.from_records(values, columns=["shape_1", "rate_1", "shape_2", "rate_2", "weight"])

    @pytest.fixture
    def irf_model(self):
        """Impulse response model object."""
        return TwoGammaImpulse(self.duration, self.offset, self.resolution)

    def test_num_frames(self, irf_model: TwoGammaImpulse):
        """Test that property num_frames is correct."""
        assert irf_model.num_frames == int(self.duration / self.resolution)

    def test_frames(self, irf_model: TwoGammaImpulse):
        """Test that property frames has correct shape."""
        assert irf_model.frames.shape == (1, irf_model.num_frames)

    @parametrize_dtype
    def test_call(self, irf_model: TwoGammaImpulse, parameters: pd.DataFrame, dtype: str):
        """Test that model response has correct shape."""
        resp = irf_model(parameters, dtype)

        assert resp.shape == (parameters.shape[0], irf_model.frames.shape[1])
