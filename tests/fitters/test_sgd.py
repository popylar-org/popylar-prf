"""Tests for stochastic gradient descent fitting."""

import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters.sgd import SGDFitter
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import Stimulus


class TestSGDFitter:
    """Tests for SGDFitter class."""

    width = 5
    height = 4
    num_frames = 10

    @pytest.fixture
    def stimulus(self):
        """Stimulus object."""
        y = np.linspace(-2, 2, num=self.height)
        x = np.linspace(-2, 2, num=self.width)
        xv, yv = np.meshgrid(x, y)
        grid = np.stack((xv, yv), axis=-1)

        design = np.ones((self.num_frames, self.height, self.width))

        return Stimulus(
            design=design,
            grid=grid,
            dimension_labels=["y", "x"],
        )

    @pytest.fixture
    def model(self):
        """Gaussian 2D PRF model instance."""
        return Gaussian2DPRFModel()

    @pytest.fixture
    def optimizer(self):
        """Optimizer instance."""
        return keras.optimizers.Adam()

    @pytest.fixture
    def loss(self):
        """Loss instance."""
        return keras.losses.MeanSquaredError()

    @pytest.fixture
    def params(self):
        """Parameters dataframe."""
        # 3 batches
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
                "shape_1": [6.0, 7.0, 5.0],
                "rate_1": [0.9, 1.0, 0.8],
                "shape_2": [7.0, 6.0, 5.0],
                "rate_2": [0.9, 1.0, 0.8],
                "weight": [0.35, 0.25, 0.45],
                "baseline": [0.0, 0.1, 0.2],
                "amplitude": [1.1, 1.0, 0.9],
            },
        )

    def test_fit(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        optimizer: keras.optimizers.Optimizer,
        loss: keras.losses.Loss,
        params: pd.DataFrame,
    ):
        """Test that fit returns parameters with the correct shape."""
        fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            optimizer=optimizer,
            loss=loss,
        )

        observed = model(stimulus, params)

        loss, sgd_params = fitter.fit(observed, params, num_steps=10)

        assert isinstance(loss, dict)
        assert isinstance(sgd_params, pd.DataFrame)
        assert sgd_params.shape == params.shape

    def test_fit_fixed_params(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        optimizer: keras.optimizers.Optimizer,
        loss: keras.losses.Loss,
        params: pd.DataFrame,
    ):
        """Test that fit with fixed parameters returns parameters with the correct shape and fixed values."""
        fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            optimizer=optimizer,
            loss=loss,
        )

        observed = model(stimulus, params)

        fixed_params = ["baseline", "amplitude"]

        loss, sgd_params = fitter.fit(observed, params, fixed_parameters=fixed_params, num_steps=10)

        assert isinstance(loss, dict)
        assert isinstance(sgd_params, pd.DataFrame)
        assert sgd_params.shape == params.shape
        assert np.all(sgd_params[fixed_params] == params[fixed_params])
