"""Impulse response models."""

import pandas as pd
from keras import ops
from prfmodel.backend import gammaln
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from .base import BaseImpulse
from .base import BatchDimensionError


def convolve_prf_impulse_response(prf_response: Tensor, impulse_response: Tensor) -> Tensor:
    """
    Convolve the encoded response from a population receptive field model with an impulse response.

    Both responses must have the same number of batches but can have different numbers of frames.

    Parameters
    ----------
    prf_response : Tensor
        Encoded population receptive field model response. Must have shape (num_batches, num_response_frames).
    impulse_response : Tensor
        Impulse response. Must have shape (num_batches, num_impulse_frames).

    Returns
    -------
    Tensor
        Convolved response with shape (num_batches, num_response_frames).

    Raises
    ------
    BatchDimensionError
        If `prf_response` and `impulse_response` have batch (first) dimensions with different sizes.

    """
    prf_response = ops.convert_to_tensor(prf_response)
    impulse_response = ops.convert_to_tensor(impulse_response)

    if prf_response.shape[0] != impulse_response.shape[0]:
        raise BatchDimensionError(
            arg_names=("prf_response", "irf_response"),
            arg_shapes=(prf_response.shape, impulse_response.shape),
        )

    prf_response_transposed = ops.expand_dims(ops.transpose(prf_response), 0)
    impulse_response_transposed = ops.expand_dims(ops.transpose(impulse_response), -1)

    response_conv = ops.depthwise_conv(prf_response_transposed, impulse_response_transposed, padding="same")

    return ops.transpose(response_conv[0, :, :])


def gamma_density(value: Tensor, shape: Tensor, rate: Tensor) -> Tensor:
    """
    Calculate the density of a gamma distribution.

    The distribution uses a shape and rate parameterization.
    Raises an error when evaluated at negative values.

    Parameters
    ----------
    value : Tensor
        The values at which to evaluate the gamma distribution. Must be > 0.
    shape : Tensor
        The shape parameter. Must be > 0.
    rate : Tensor
        The rate parameter. Must be > 0.

    Returns
    -------
    Tensor
        The density of the gamma distribution at `value`.

    Raises
    ------
    ValueError
        If `values`, `shape`, or `rate` are zero or negative.

    """
    if not ops.all(value > 0.0):
        msg = "Values must be > 0"
        raise ValueError(msg)

    if not ops.all(shape > 0.0):
        msg = "Shape parameters must be > 0"
        raise ValueError(msg)

    if not ops.all(rate > 0.0):
        msg = "Rate parameters must be > 0"
        raise ValueError(msg)

    # Calculate log density and then exponentiate
    return ops.exp(shape * ops.log(rate) + (shape - 1) * ops.log(value) - rate * value - gammaln(shape))


class TwoGammaImpulse(BaseImpulse):
    """
    Weighted sum of two gamma distributions impulse response model.

    Predicts an impulse response that is the weighted sum of two gamma distributions.
    The model has five parameters: `shape_1` and `rate_1` for the first, `shape_2` and `rate_2` for the second
    gamma distribution, and `weight` for the relative weight of the first gamma distribution. The model prediction is:
    `p(x) = weight * p(shape_1, rate_1) + (1 - weight) * p(shape_2, rate_2)`.

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

    Examples
    --------
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "shape_1": [2.0, 1.0, 1.5],
    >>>     "rate_1": [1.0, 1.0, 1.0],
    >>>     "shape_2": [1.5, 2.0, 1.0],
    >>>     "rate_2": [1.0, 1.0, 1.0],
    >>>     "weight": [0.7, 0.2, 0.5],
    >>> })
    >>> impulse_model = TwoGammaImpulse(
    >>>     duration=100.0 # 100 seconds
    >>> )
    >>> resp = impulse_model(params)
    >>> print(resp.shape) # (num_rows, duration)
    (3, 100)

    """

    def __init__(self, duration: float = 32.0, offset: float = 32.0, resolution: float = 1.0, dtype: str = "float64"):
        super().__init__(duration, offset, resolution, dtype)

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: `shape_1`, `rate_1`, `shape_2`, `rate_2`, `weight`."""
        return ["shape_1", "rate_1", "shape_2", "rate_2", "weight"]

    def __call__(self, parameters: pd.DataFrame) -> Tensor:
        """
        Predict the model response.

        Parameters
        ----------
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different batches. Must contain the columns `shape_1`, `rate_1`, `shape_2`, `rate_2`, and `weight`.

        """
        shape_1 = convert_parameters_to_tensor(parameters[["shape_1"]])
        rate_1 = convert_parameters_to_tensor(parameters[["rate_1"]])
        shape_2 = convert_parameters_to_tensor(parameters[["shape_2"]])
        rate_2 = convert_parameters_to_tensor(parameters[["rate_2"]])
        weight = convert_parameters_to_tensor(parameters[["weight"]])
        dens_1 = gamma_density(self.frames, shape_1, rate_1)
        dens_2 = gamma_density(self.frames, shape_2, rate_2)
        return weight * dens_1 + (1.0 - weight) * dens_2
