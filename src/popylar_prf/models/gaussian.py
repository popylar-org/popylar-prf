"""Gaussian population receptive field response models."""

from keras import ops  # type: ignore[import-untyped]
from popylar_prf.stimulus import GridDimensionsError
from popylar_prf.typing import Tensor
from .base import ParameterShapeError
from .base import ParameterSizeError

_MIN_PARAMETER_DIM = 2


class GridMuDimensionsError(Exception):
    """
    Exception raised when the dimensions of the stimulus grid and the Gaussian mu parameter do not match.

    Parameters
    ----------
    grid_shape : tuple of int
        Shape of the stimulus grid.
    mu_shape : tuple of int
        Shape of the Gaussian mu parameter.

    """

    def __init__(self, grid_shape: tuple[int, ...], mu_shape: tuple[int, ...]):
        super().__init__(f"Dimensions of 'grid' {grid_shape} and 'mu' {mu_shape} do not match")


def predict_gaussian_response(grid: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """
    Predict a isotropic Gaussian population receptive field response.

    The dimensionality of the Gaussian depends on the number of dimensions of `grid` and `mu`.

    Parameters
    ----------
    grid : Tensor
        Stimulus grid for which to make predictions.
    mu : Tensor
        Centroid of the population receptive field. The first axis corresponds to the number of voxels. The second axis
        corresponds to the number of grid dimensions and must match the size of the last `grid` axis.
    sigma : Tensor
        Size of the population receptive field. The first axis corresponds to the number of voxels and the size must
        match the size of the first `mu` axis.

    Returns
    -------
    Tensor
        The predicted Gaussian population receptive field response with shape (num_voxels, ...)
        where `...` corresponds to the dimensions of the Gaussian.

    Raises
    ------
    GridDimensionsError
        If the grid has mismatching dimensions.
    ParameterShapeError
        If `mu` or `sigma` have less than two dimensions.
    ParameterSizeError
        If `mu` and `sigma` have first dimensions with different sizes.
    GridMuDimensionsError
        If the grid and mu dimensions do not match.
    ParameterSizeError
        If the first axes of `mu` and `sigma` do not have the same size.

    Examples
    --------
    Predict a 2D Gaussian response:

    >>> import numpy as np
    >>> # Define a 2D grid
    >>> num_x, num_y = 10, 10
    >>> x = np.linspace(-3, 3, num_x)
    >>> y = np.linspace(-4, 4, num_y)
    >>> xv, yv = np.meshgrid(x, y)
    >>> grid = np.stack((xv, yv), axis=-1)  # shape (10, 10, 2)
    >>> # Define 2D centroids of Gaussian for 3 voxels
    >>> mu = np.array([ # shape (3, 2), first column y, second column x
    >>>     [0.0, 1.0],
    >>>     [1.0, 0.0],
    >>>     [0.0, 0.0],
    >>> ])
    >>> # Define size of Gaussian for 3 voxels
    >>> sigma = np.array([[1.0], [1.5], [2.0]]) # shape (3, 1)
    >>> resp = predict_gaussian_response(grid, mu, sigma)
    >>> print(resp.shape) # (num_voxels, num_y, num_x)
    (3, 10, 10)

    """
    if not len(grid.shape[:-1]) == grid.shape[-1]:
        raise GridDimensionsError(grid.shape)

    if len(mu.shape) < _MIN_PARAMETER_DIM:
        raise ParameterShapeError(
            parameter_name="mu",
            parameter_shape=mu.shape,
        )

    if len(sigma.shape) < _MIN_PARAMETER_DIM:
        raise ParameterShapeError(
            parameter_name="sigma",
            parameter_shape=sigma.shape,
        )

    if grid.shape[-1] != mu.shape[-1]:
        raise GridMuDimensionsError(grid.shape, mu.shape)

    if mu.shape[0] != sigma.shape[0]:
        raise ParameterSizeError(
            parameter_names=("mu", "sigma"),
            parameter_shapes=(mu.shape, sigma.shape),
        )

    grid = ops.convert_to_tensor(grid)
    mu = ops.convert_to_tensor(mu)
    sigma = ops.convert_to_tensor(sigma)

    # Expand mu to same shape as grid: (num_voxels, ..., grid.shape[-1])
    mu_expand = tuple(range(1, grid.shape[-1] + 1))
    # Expand sigma to same shape as grid but omit last grid dimension
    sigma_expand = mu_expand[:-1]

    mu = ops.expand_dims(mu, axis=mu_expand)
    sigma = ops.expand_dims(sigma, axis=sigma_expand)
    # Add new first dimension to grid corresponding to num_voxels
    grid = ops.expand_dims(grid, axis=0)

    # Gaussian response
    resp = ops.sum(ops.square(grid - mu), axis=-1)
    resp /= 2 * ops.square(sigma)

    return ops.exp(-resp)
