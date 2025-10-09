"""Stimulus encoding classes."""

from keras import ops
from prfmodel.typing import Tensor


class ResponseDesignShapeError(Exception):
    """
    Exception raised when the shapes of the model response and stimulus design do not match.

    Both must have the same shape after the first dimension.

    Parameters
    ----------
    response_shape : tuple of int
        Shape of the model response array.
    design_shape : tuple of int
        Shape of the design array.
    """

    def __init__(self, response_shape: tuple[int, ...], design_shape: tuple[int, ...]):
        super().__init__(f"Shapes of 'response' {response_shape} and 'design' {design_shape} do not match")


class Encoder:
    """Stimulus encoder.

    Predicts the stimulus-encoded response of a model by multiplying the stimulus design with the response along the
    stimulus dimensions and summing over them.

    Examples
    --------
    Encode a 1D model response:

    >>> import numpy as np
    >>> num_batches = 3
    >>> num_frames = 10
    >>> height = 5
    >>> # Create a dummy stimulus design
    >>> design = np.ones((num_frames, height))
    >>> # Create a dummy model response that varies with the height of a stimulus grid
    >>> resp = np.ones((num_batches, height)) * np.expand_dims(np.sin(np.arange(height)), 0)
    >>> print(resp.shape) # (num_batches, height)
    (3, 5)
    >>> # Create an encoder instance
    >>> encoder = Encoder()
    >>> resp_encoded = encoder(resp, design)
    >>> print(resp_encoded.shape) (num_batches, num_frames)
    (3, 10)

    Encode a 2D model response:

    >>> # Add width dimension
    >>> width = 4
    >>> # Create a dummy stimulus design
    >>> design = np.ones((num_frames, height, width))
    >>> # Create a dummy model response that varies with the width of a stimulus grid
    >>> resp = np.ones((num_batches, height, width)) * np.expand_dims(np.sin(np.arange(width)), (0, 1))
    >>> print(resp.shape) # (num_batches, height, width)
    (3, 5, 4)
    >>> # Create an encoder instance
    >>> encoder = Encoder()
    >>> resp_encoded = encoder(resp, design)
    >>> print(resp_encoded.shape) (num_batches, num_frames)
    (3, 10)


    """

    def __call__(self, response: Tensor, design: Tensor) -> Tensor:
        """Predict the stimulus-encoded model response.

        Parameters
        ----------
        response : Tensor
            The model response. The first dimension corresponds to the number of batches.
            Additional dimensions correspond to the stimulus dimensions.
        design : Tensor
            The stimulus design containing the stimulus value in one or more dimensions over different time frames.
            The first axis is assumed to be time frames. Additional axes represent stimulus dimensions.

        Returns
        -------
        Tensor
            The stimulus encoded model response with two dimensions.
            The first dimension is the batch dimension, the second dimension is time frames.

        """
        response = ops.convert_to_tensor(response)
        design = ops.convert_to_tensor(design)

        if response.shape[1:] != design.shape[1:]:
            raise ResponseDesignShapeError(response.shape, design.shape)

        design = ops.expand_dims(design, 0)
        response = ops.expand_dims(response, 1)
        x = response * design
        axes = ops.arange(2, 2 + len(design.shape) - 2)
        return ops.sum(x, axis=axes)
