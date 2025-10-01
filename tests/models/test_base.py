"""Test model base classes."""

import numpy as np
import pandas as pd
import pytest
from popylar_prf.models.base import BaseModel
from popylar_prf.models.base import ParameterShapeError
from popylar_prf.models.base import ParameterSizeError
from popylar_prf.models.base import ResponseModel
from popylar_prf.stimulus import Stimulus


def test_parameter_shape_error():
    """Test that ParameterShapeError shows correct parameter name and shape in error message."""
    param_name = "param_1"
    param_shape = 1

    with pytest.raises(ParameterShapeError) as excinfo:
        raise ParameterShapeError(param_name, param_shape)

    assert param_name in str(excinfo.value)
    assert str(param_shape) in str(excinfo.value)


def test_parameter_size_error():
    """Test that ParameterSizeError shows correct parameter names and shapes in error message."""
    param_names = ("param_1", "param_2")
    param_shapes = ((2, 1), (1, 1))

    with pytest.raises(ParameterSizeError) as excinfo:
        raise ParameterSizeError(param_names, param_shapes)

    for param_name, param_shape in zip(param_names, param_shapes, strict=False):
        assert param_name in str(excinfo.value)
        assert str(param_shape[0]) in str(excinfo.value)


class TestBaseModel:
    """Tests for BaseModel class."""

    model_class = BaseModel

    def test_abstract_fail(self):
        """Test that model instantiation fails."""
        with pytest.raises(TypeError):
            _ = self.model_class()

    def test_parameter_names(self):
        """Test that parameter_names property exists with correct signature."""

        # Create a local subclass from which we can remove the abstract methods
        # without modifying the global class
        class LocalClass(self.model_class):
            pass

        # Remove abstract methods to init the base class
        LocalClass.__abstractmethods__ = set()
        model = LocalClass()
        # Check that the 'parameter_names' property exists with the correct signature
        param_names = model.parameter_names
        assert param_names == []


# Inherit all checks from TestBaseModel
class TestResponseModel(TestBaseModel):
    """Tests for ResponseModel class."""

    model_class = ResponseModel

    def test_predict(self):
        """Test that predict method exists with correct signature."""

        class LocalClass(self.model_class):
            pass

        LocalClass.__abstractmethods__ = set()
        model = LocalClass()

        design = np.zeros((1, 2, 1))
        grid = np.ones((2, 1, 2))

        stimulus = Stimulus(
            dimension_labels=["x", "y"],
            design=design,
            grid=grid,
        )

        # Empty parameters dataframe
        params = pd.DataFrame()
        # Check that the '__call__' method exists with the correct signature
        preds = np.asarray(model(stimulus, params))

        assert np.all(preds == 0.0)
