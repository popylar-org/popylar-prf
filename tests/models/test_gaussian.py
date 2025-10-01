"""Test Gaussian model classes."""

import numpy as np
import pytest
from popylar_prf.models.base import ParameterBatchDimensionError
from popylar_prf.models.base import ParameterShapeError
from popylar_prf.models.gaussian import GridMuDimensionsError
from popylar_prf.models.gaussian import _check_gaussian_args
from popylar_prf.models.gaussian import _expand_gaussian_args
from popylar_prf.models.gaussian import predict_gaussian_response
from popylar_prf.stimulus import GridDimensionsError


class TestCheckGaussianArgs:
    """Tests for _check_gaussian_args function."""

    def test_grid_dimensions_error(self):
        """Test that GridDimensionsError is raised."""
        grid = np.ones((4, 5, 1))  # len(shape[:-1]) = 2, shape[-1] = 1
        mu = np.ones((3, 1))
        sigma = np.ones((3, 1))
        with pytest.raises(GridDimensionsError):
            _check_gaussian_args(grid, mu, sigma)

    def test_grid_mu_dimensions_error(self):
        """Test that GridMuDimensionsError is raised."""
        grid = np.ones((4, 5, 2))
        mu = np.ones((3, 3))  # mu.shape[-1] = 3, grid.shape[-1] = 2
        sigma = np.ones((3, 1))
        with pytest.raises(GridMuDimensionsError):
            _check_gaussian_args(grid, mu, sigma)

    def test_parameter_size_error(self):
        """Test that ParameterSizeError is raised."""
        grid = np.ones((4, 5, 2))
        mu = np.ones((2, 2))
        sigma = np.ones((3, 1))  # Mismatch in first axis
        with pytest.raises(ParameterBatchDimensionError):
            _check_gaussian_args(grid, mu, sigma)

    def test_parameter_shape_error(self):
        """Test that ParameterShapeError is raised."""
        grid = np.ones((4, 1))
        mu = np.ones(1)  # Less than two dimensions
        sigma = np.ones((3, 1))
        with pytest.raises(ParameterShapeError):
            _check_gaussian_args(grid, mu, sigma)

        mu = np.ones((3, 1))
        sigma = np.ones(3)  # Less than two dimensions

        with pytest.raises(ParameterShapeError):
            _check_gaussian_args(grid, mu, sigma)


class TestSetup:
    """Setup parameters and objects for testing."""

    width: int = 5
    height: int = 4
    depth: int = 3

    @pytest.fixture
    def grid_1d(self):
        """1D stimulus grid."""
        return np.expand_dims(np.linspace(-2, 2, num=self.height), axis=1)  # (height, 1)

    @pytest.fixture
    def grid_2d(self):
        """2D stimulus grid."""
        y = np.linspace(-2, 2, num=self.height)
        x = np.linspace(-2, 2, num=self.width)
        xv, yv = np.meshgrid(x, y)
        return np.stack((xv, yv), axis=-1)  # (height, width, 2)

    @pytest.fixture
    def grid_3d(self):
        """3D stimulus grid."""
        y = np.linspace(-2, 2, num=self.height)
        x = np.linspace(-2, 2, num=self.width)
        z = np.linspace(-2, 2, num=self.depth)
        xv, yv, zv = np.meshgrid(x, y, z)
        return np.stack((xv, yv, zv), axis=-1)  # (height, width, depth, 3)

    @pytest.fixture
    def mu_1d(self):
        """1D Gaussian mu parameters."""
        return np.expand_dims(np.array([0.0, 1.0, 2.0]), axis=1)  # (num_voxels, 1)

    @pytest.fixture
    def mu_2d(self):
        """2D Gaussian mu parameters."""
        return np.array(
            [  # (num_voxels, 2)
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
        )

    @pytest.fixture
    def mu_3d(self):
        """3D Gaussian mu parameters."""
        return np.array(
            [  # (num_voxels, 3)
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
        )

    @pytest.fixture
    def sigma(self):
        """Gaussian sigma parameters."""
        return np.expand_dims(np.array([1.0, 1.5, 2.0]), axis=1)  # (num_voxels, 1)


class TestExpandGaussianArgs(TestSetup):
    """Tests for _expand_gaussian_args function."""

    @staticmethod
    def _check_shapes(grid: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        assert len(grid.shape) == len(mu.shape)
        assert len(mu.shape) - 1 == len(sigma.shape)
        assert grid.shape[-1] == mu.shape[-1]

    def test_expand_gaussian_args_1d(self, grid_1d: np.ndarray, mu_1d: np.ndarray, sigma: np.ndarray):
        """Test that 1D args are correctly expanded."""
        grid, mu, sigma = _expand_gaussian_args(grid_1d, mu_1d, sigma)

        self._check_shapes(grid, mu, sigma)

    def test_expand_gaussian_args_2d(self, grid_2d: np.ndarray, mu_2d: np.ndarray, sigma: np.ndarray):
        """Test that 2D args are correctly expanded."""
        grid, mu, sigma = _expand_gaussian_args(grid_2d, mu_2d, sigma)

        self._check_shapes(grid, mu, sigma)

    def test_expand_gaussian_args_3d(self, grid_3d: np.ndarray, mu_3d: np.ndarray, sigma: np.ndarray):
        """Test that 3D args are correctly expanded."""
        grid, mu, sigma = _expand_gaussian_args(grid_3d, mu_3d, sigma)

        self._check_shapes(grid, mu, sigma)


class TestPredictGaussianResponse(TestSetup):
    """Tests for predict_gaussian_response function."""

    def test_predict_gaussian_response_1d(self, grid_1d: np.ndarray, mu_1d: np.ndarray, sigma: np.ndarray):
        """Test that 1D response prediction returns correct shape."""
        preds = predict_gaussian_response(grid_1d, mu_1d, sigma)

        assert preds.shape == (3, self.height)

    def test_predict_gaussian_response_2d(self, grid_2d: np.ndarray, mu_2d: np.ndarray, sigma: np.ndarray):
        """Test that 2D response prediction returns correct shape."""
        preds = predict_gaussian_response(grid_2d, mu_2d, sigma)

        assert preds.shape == (3, self.height, self.width)

    def test_predict_gaussian_response_3d(self, grid_3d: np.ndarray, mu_3d: np.ndarray, sigma: np.ndarray):
        """Test that 3D response prediction returns correct shape."""
        preds = predict_gaussian_response(grid_3d, mu_3d, sigma)

        assert preds.shape == (3, self.height, self.width, self.depth)
