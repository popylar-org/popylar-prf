"""Model base classes."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
import pandas as pd
from keras import ops
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor

_MIN_PARAMETER_DIM = 2


class BatchDimensionError(Exception):
    """
    Exception raised when arguments have different sizes in the batch (first) dimension.

    Parameters
    ----------
    arg_names: Sequence[str]
        Names of arguments that have different sizes in batch dimension.
    arg_shapes: Sequence[tuple of int]
        Shapes of arguments that have different sizes in batch dimension.

    """

    def __init__(self, arg_names: Sequence[str], arg_shapes: Sequence[tuple[int, ...]]):
        names = ", ".join(arg_names)
        shapes = ", ".join([str(s[0]) for s in arg_shapes])

        super().__init__(f"Arguments {names} have different sizes in batch (first) dimension: {shapes}")


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
    def parameter_names(self) -> list[str]:
        """A list with names of parameters that are used by the model."""


class BasePRFResponse(BaseModel):
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


class BaseImpulse(BaseModel):
    """
    Abstract base class for impulse response models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom impulse response models.
    Subclasses must override the abstract `__call__` method.

    #TODO: Link to Example on how to create custom impulse response models.

    Parameters
    ----------
    duration : float, default=32.0
        The duration of the impulse response (in seconds).
    offset : float, default=0.0001
        The offset of the impulse response (in seconds). By default a very small offset is added to prevent infinite
        response values at t = 0.
    resolution : float, default=1.0
        The time resultion of the impulse response (in seconds), that is the number of points per second at which the
        impulse response function is evaluated.
    dtype : str, default="float64"
        Dtype of the impulse response.

    """

    def __init__(
        self,
        duration: float = 32.0,
        offset: float = 0.0001,
        resolution: float = 1.0,
        dtype: str = "float64",
    ):
        super().__init__()

        self.duration = duration
        self.offset = offset
        self.resolution = resolution
        self.dtype = dtype

    @property
    def num_frames(self) -> int:
        """The total number of time frames at which the impulse response function is evaluated."""
        return int(self.duration / self.resolution)

    @property
    def frames(self) -> Tensor:
        """
        The time frames at which the impulse response function is evaluated.

        Time frames are linearly interpolated between `offset` and `duration` and have shape (1, `num_frames`).

        """
        return ops.expand_dims(ops.linspace(self.offset, self.duration, self.num_frames, dtype=self.dtype), 0)

    @abstractmethod
    def __call__(self, parameters: pd.DataFrame) -> Tensor:
        """
        Compute the impulse response.

        Parameters
        ----------
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        Tensor
            Model predictions of shape `(num_voxels, num_frames)`. The number of voxels is the number of rows in
            `parameters`.

        """
