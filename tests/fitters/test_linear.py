"""Tests for linear fitting."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters.linear import LeastSquaresFitter
from prfmodel.fitters.linear import LeastSquaresHistory
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import Stimulus
from .conftest import TestSetup
from .conftest import parametrize_dtype


class TestLeastSquaresFitter(TestSetup):
    """Tests for GridFitter class."""

    @pytest.mark.parametrize("target_parameters", [[], ["a", "b", "c"]])
    def test_fit_target_parameters_value_error(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        target_parameters: tuple[str],
    ):
        """Test that 'target_parameters' with incorrect length raises error."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
        )

        observed = model(stimulus, params)

        with pytest.raises(ValueError):
            _ = fitter.fit(observed, params, target_parameters=target_parameters)

    @pytest.mark.parametrize("target_parameters", [None, ("a", "b", "c")])
    def test_fit_target_parameters_type_error(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        target_parameters: tuple[str],
    ):
        """Test that 'target_parameters' with incorrect input type raises error."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
        )

        observed = model(stimulus, params)

        with pytest.raises(TypeError):
            _ = fitter.fit(observed, params, target_parameters=target_parameters)

    @parametrize_dtype
    @pytest.mark.parametrize("target_parameters", [["amplitude"], ["baseline"], ["amplitude", "baseline"]])
    def test_fit(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        target_parameters: tuple[str],
        dtype: str,
    ):
        """Test that fit returns parameters with the correct shape."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
            dtype=dtype,
        )

        observed = model(stimulus, params)

        history, ls_params = fitter.fit(observed, params, target_parameters=target_parameters)

        assert isinstance(history, LeastSquaresHistory)
        assert isinstance(history.history, dict)
        assert isinstance(history.history["loss"], np.ndarray)
        assert isinstance(ls_params, pd.DataFrame)
        assert ls_params.shape == params.shape
