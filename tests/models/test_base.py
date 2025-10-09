"""Test model base classes."""

import pytest
from prfmodel.models.base import BaseModel
from prfmodel.models.base import BatchDimensionError
from prfmodel.models.base import ParameterShapeError
from prfmodel.models.base import ResponseModel


def test_parameter_shape_error():
    """Test that ParameterShapeError shows correct parameter name and shape in error message."""
    param_name = "param_1"
    param_shape = 1

    with pytest.raises(ParameterShapeError) as excinfo:
        raise ParameterShapeError(param_name, param_shape)

    assert param_name in str(excinfo.value)
    assert str(param_shape) in str(excinfo.value)


def test_batch_dimension_error():
    """Test that BatchDimensionError shows correct arg names and shapes in error message."""
    arg_names = ("arg_1", "arg_2")
    arg_shapes = ((2, 1), (1, 1))

    with pytest.raises(BatchDimensionError) as excinfo:
        raise BatchDimensionError(arg_names, arg_shapes)

    for arg_name, arg_shape in zip(arg_names, arg_shapes, strict=False):
        assert arg_name in str(excinfo.value)
        assert str(arg_shape[0]) in str(excinfo.value)


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
