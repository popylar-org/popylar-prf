"""Test model base classes."""

import pytest
from popylar_prf.models.base import BaseModel
from popylar_prf.models.base import ParameterBatchDimensionError
from popylar_prf.models.base import ParameterShapeError
from popylar_prf.models.base import ResponseModel


def test_parameter_shape_error():
    """Test that ParameterShapeError shows correct parameter name and shape in error message."""
    param_name = "param_1"
    param_shape = 1

    with pytest.raises(ParameterShapeError) as excinfo:
        raise ParameterShapeError(param_name, param_shape)

    assert param_name in str(excinfo.value)
    assert str(param_shape) in str(excinfo.value)


def test_parameter_batch_dimension_error():
    """Test that ParameterBatchDimensionError shows correct parameter names and shapes in error message."""
    param_names = ("param_1", "param_2")
    param_shapes = ((2, 1), (1, 1))

    with pytest.raises(ParameterBatchDimensionError) as excinfo:
        raise ParameterBatchDimensionError(param_names, param_shapes)

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


# Inherit all checks from TestBaseModel
class TestResponseModel(TestBaseModel):
    """Tests for ResponseModel class."""

    model_class = ResponseModel
