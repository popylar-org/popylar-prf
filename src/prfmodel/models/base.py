"""Model base classes."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
import pandas as pd
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor

_MIN_PARAMETER_DIM = 2


class ParameterBatchDimensionError(Exception):
    """
    Exception raised when model parameters have different sizes in the batch (first) dimension.

    Parameters
    ----------
    parameter_names: Sequence[str]
        Names of parameters that have different sizes in batch dimension.
    parameter_shapes: Sequence[tuple of int]
        Shapes of parameters that have different sizes in batch dimension.

    """

    def __init__(self, parameter_names: Sequence[str], parameter_shapes: Sequence[tuple[int, ...]]):
        names = ", ".join(parameter_names)
        shapes = ", ".join([str(s[0]) for s in parameter_shapes])

        super().__init__(f"Parameters {names} have different sizes in batch (first) dimension: {shapes}")


class ParameterShapeError(Exception):
    """
    Exception raised when a model parameter has less than two dimensions.

    Parameters
    ----------
    parameter_name: str
        Parameter name.
    parameter_shape: tuple of int
        Parameter shape.

    """

    def __init__(self, parameter_name: str, parameter_shape: tuple[int, ...]):
        super().__init__(
            f"Parameter {parameter_name} must have at least two dimensions but has shape {parameter_shape}",
        )


class BaseModel(ABC):
    """
    Abstract base class for models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom model classes.
    Subclasses must override the abstract `parameter_names` property.

    Attributes
    ----------
    parameter_names

    Examples
    --------
    Create a custom model class that inherits from the base class:

    >>> class CustomModel(BaseModel):
    >>>     @property
    >>>     def parameter_names(self):
    >>>         return ["a", "b"]
    >>> model = CustomModel()
    >>> print(model.parameter_names)
    ["a", "b"]

    """

    @property
    @abstractmethod
    def parameter_names(self) -> list:
        """A list with names of parameters that are used by the model."""


class ResponseModel(BaseModel):
    """
    Abstract base class for population receptive field response models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom population receptive field models.
    Subclasses must override the abstract `__call__` method.

    #TODO: Link to Example on how to create custom response models.

    """

    @abstractmethod
    def __call__(self, stimulus: Stimulus, parameters: pd.DataFrame) -> Tensor:
        """
        Predict the model response for a stimulus.

        Parameters
        ----------
        stimulus : Stimulus
            Stimulus object.
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        Tensor
            Model predictions of shape `(num_voxels, ...)`. The number of voxels is the number of rows in
            `parameters`. The number and size of other axes depends on the stimulus.

        """
