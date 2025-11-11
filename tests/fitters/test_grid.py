"""Tests for grid fitting."""

import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters.grid import GridFitter
from prfmodel.fitters.grid import GridHistory
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import Stimulus
from .conftest import TestSetup
from .conftest import parametrize_dtype


class TestGridFitter(TestSetup):
    """Tests for GridFitter class."""

    @pytest.fixture
    def param_ranges(self):
        """Parameter ranges."""
        return {
            "mu_x": [-1.0, 0.0, 1.0],
            "mu_y": [1.0, 0.0],
            "sigma": [1.0, 2.0],
            "shape_1": [5.0, 6.0],
            "rate_1": [0.9, 1.0],
            "shape_2": [12.0],
            "rate_2": [0.9],
            "weight": [0.35],
            "baseline": [0.0],
            "amplitude": [1.0],
        }

    @parametrize_dtype
    @pytest.mark.parametrize("loss", [None, keras.losses.MeanSquaredError(reduction="none")])
    def test_fit(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        loss: keras.losses.Loss,
        params: pd.DataFrame,
        param_ranges: dict[str, np.ndarray],
        dtype: str,
    ):
        """Test that fit returns parameters with the correct shape."""
        fitter = GridFitter(
            model=model,
            stimulus=stimulus,
            loss=loss,
            dtype=dtype,
        )

        observed = model(stimulus, params)

        history, grid_params = fitter.fit(observed, param_ranges, chunk_size=2)

        assert isinstance(history, GridHistory)
        assert isinstance(history.history, dict)
        assert isinstance(history.history["loss"], np.ndarray)
        assert isinstance(grid_params, pd.DataFrame)
        assert grid_params.shape == params.shape
