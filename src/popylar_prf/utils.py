"""Utility functions."""

import pandas as pd
from keras import ops
from .typing import Tensor


def convert_parameters_to_tensor(parameters: pd.DataFrame) -> Tensor:
    """Convert model parameters in a dataframe into a tensor.

    Parameters
    ----------
    parameters : pandas.DataFrame
        Dataframe with columns containing different model parameters and rows containing
        parameter values for different voxels.

    Returns
    -------
    Tensor
        Tensor with the first axis corresponding to voxels and the second axis corresponding to different parameters.

    Examples
    --------
    Single parameters:

    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "param_1": [0.0, 1.0, 2.0],
    >>> })
    >>> x = convert_parameters_to_tensor(params)
    >>> print(x.shape)
    (3, 1)

    Multiple parameters:

    >>> params = pd.DataFrame({
    >>>     "param_1": [0.0, 1.0, 2.0],
    >>>     "param_2": [0.0, -1.0, -2.0],
    >>> })
    >>> x = covert_parameters_to_tensor(params)
    >>> print(x.shape)
    (3, 2)

    """
    return ops.convert_to_tensor(parameters)
