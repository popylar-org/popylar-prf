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

    Notes
    -----
    Before convolving both responses, the `prf_response` is padded on the left side in the
    `num_frames` dimension by repeating the first value of each batch. This ensures that the output of the convolution
    has the same shape as `prf_response` and the `impulse_response` starts at every frame of the `prf_response`.

    Raises
    ------
    BatchDimensionError
        If `prf_response` and `impulse_response` have batch (first) dimensions with different sizes.

    """
    prf_response = ops.convert_to_tensor(prf_response)
    impulse_response = ops.flip(ops.convert_to_tensor(impulse_response), 1)

    if prf_response.shape[0] != impulse_response.shape[0]:
        raise BatchDimensionError(
            arg_names=("prf_response", "impulse_response"),
            arg_shapes=(prf_response.shape, impulse_response.shape),
        )

    # We pad the pRF response signal on the left side by repeating the first response value
    # This ensures that, during the convolution, the impulse response starts at every frame of the pRF response
    # and the output shape is the same as the pRF response shape
    pad_len = impulse_response.shape[1] - 1
    response_padding = ops.tile(prf_response[:, :1], (1, pad_len))
    prf_response_padded = ops.concatenate([response_padding, prf_response], axis=1)

    # Transpose to meet shape requirements of depthwise convolution
    prf_response_transposed = ops.expand_dims(ops.transpose(prf_response_padded), 0)
    impulse_response_transposed = ops.expand_dims(ops.transpose(impulse_response), -1)

    # We perform 1D depthwise convolution
    response_conv = ops.depthwise_conv(
        prf_response_transposed,
        ops.flip(impulse_response_transposed, axis=1),  # Flip along time axis for convolution
        padding="valid",
    )

    # Transpose back and remove first dummy dimension: (num_batches, num_frames)
    return ops.transpose(response_conv[0, :, :])


def gamma_density(value: Tensor, shape: Tensor, rate: Tensor, norm: bool = True) -> Tensor:
    r"""
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
    norm : bool, default=True
        Whether to compute the normalized density.

    Returns
    -------
    Tensor
        The density of the gamma distribution at `value`.

    Notes
    -----
    The unnormalized density of the gamma distribution
    with `shape` :math:`\alpha` and `rate` :math:`\lambda` is given by:

    .. math::

        f(x) = x^{\mathtt{\alpha} - 1} e^{\mathtt{\lambda} x}.

    When `norm=True`, the density is multiplied with a normalizing constant:

    .. math::

        f_{norm} = \frac{\mathtt{\lambda}^{\mathtt{\alpha}}}{\Gamma(\mathtt{\alpha})} * f(x).

    Raises
    ------
    ValueError
        If `values`, `shape`, or `rate` are zero or negative.

    """
    value = ops.convert_to_tensor(value)
    shape = ops.convert_to_tensor(shape)
    rate = ops.convert_to_tensor(rate)

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
    dens = (shape - 1) * ops.log(value) - rate * value

    if norm:
        # Normalize
        return ops.exp(shape * ops.log(rate) + dens - gammaln(shape))

    return ops.exp(dens)


class TwoGammaImpulse(BaseImpulse):
    r"""
    Weighted difference of two gamma distributions impulse response model.

    Predicts an impulse response that is the weighted difference of two gamma distributions.
    The model has five parameters: `shape_1` and `rate_1` for the first, `shape_2` and `rate_2` for the second
    gamma distribution, and `weight` for the relative weight of the first gamma distribution.

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

    Notes
    -----
    The predicted impulse response at time :math:`t` with `shape_1` :math:`\alpha_1`, `rate_1` :math:`\lambda_1`,
    `shape_2` :math:`\alpha_2`, `rate_2` :math:`\lambda_2`, and `weight` :math:`w` is:

    .. math::

        f(t) = \hat{f}_{\text{gamma}}(t; \alpha_1, \lambda_1) - w \hat{f}_{\text{gamma}}(t; \alpha_2, \lambda_2)

    The gamma distributions are divided by their respective maximum, so that their highest peak has an amplitude of 1:

    .. math::
        \hat{f}_{\text{gamma}}(t; \alpha, \lambda) = \frac{f_{\text{gamma}}(t; \alpha, \lambda)}
        {\text{max}(f_{\text{gamma}}(t; \alpha, \lambda))}

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

    def __init__(self, duration: float = 32.0, offset: float = 0.0001, resolution: float = 1.0, dtype: str = "float64"):
        super().__init__(duration, offset, resolution, dtype)

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: `shape_1`, `rate_1`, `shape_2`, `rate_2`, `weight`.

        """
        return ["shape_1", "rate_1", "shape_2", "rate_2", "weight"]

    def __call__(self, parameters: pd.DataFrame) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different batches. Must contain the columns `shape_1`, `rate_1`, `shape_2`, `rate_2`, and `weight`.

        Returns
        -------
        Tensor
            The predicted impulse response with shape `(num_batches, num_frames)`.

        """
        shape_1 = convert_parameters_to_tensor(parameters[["shape_1"]])
        rate_1 = convert_parameters_to_tensor(parameters[["rate_1"]])
        shape_2 = convert_parameters_to_tensor(parameters[["shape_2"]])
        rate_2 = convert_parameters_to_tensor(parameters[["rate_2"]])
        weight = convert_parameters_to_tensor(parameters[["weight"]])
        # Compute unnormalized density because normalizing constant cancels out when taking difference anyway
        dens_1 = gamma_density(self.frames, shape_1, rate_1, norm=False)
        dens_1_norm = dens_1 / ops.max(dens_1, axis=1, keepdims=True)
        dens_2 = gamma_density(self.frames, shape_2, rate_2, norm=False)
        dens_2_norm = dens_2 / ops.max(dens_2, axis=1, keepdims=True)
        return dens_1_norm - weight * dens_2_norm
