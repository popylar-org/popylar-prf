"""Composite population receptive field models."""

import pandas as pd
from keras import ops
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from .base import BaseImpulse
from .base import BaseModel
from .base import BasePRFModel
from .base import BasePRFResponse
from .encoding import encode_prf_response
from .impulse import TwoGammaImpulse
from .impulse import convolve_prf_impulse_response
from .linear import BaselineAmplitude


class SimplePRFModel(BasePRFModel):
    """
    Simple composite population receptive field model.

    This is a generic class that combines a population receptive field, impulse, and linear response.

    Parameters
    ----------
    prf_model : BasePRFResponse
        A population receptive field response model instance.
    impulse_model : str or BaseImpulse or None, default="default", optional
        An impulse response model instance. Can also be `"default"` to use a `TwoGammaImpulse` instance with default
        values.
    linear_model : str or BaseModel or None, default="default", optional
        An linear response model instance. Can also be `"default"` to use a `BaselineAmplitude` instance with default
        values.

    Notes
    -----
    The simple composite model follows five steps:

    1. The population receptive field response model makes a prediction for the stimulus grid.
    2. The response is encoded with the stimulus design.
    3. A impulse response model generates an impulse response.
    4. The encoded response is convolved with the impulse response.
    5. The linear model modifies the convolved response.

    """

    def __init__(
        self,
        prf_model: BasePRFResponse,
        impulse_model: str | BaseImpulse | None = "default",
        linear_model: str | BaseModel | None = "default",
    ):
        if impulse_model == "default":
            impulse_model = TwoGammaImpulse()

        if linear_model == "default":
            linear_model = BaselineAmplitude()

        super().__init__(
            prf_model=prf_model,
            impulse_model=impulse_model,
            linear_model=linear_model,
        )

    def __call__(self, stimulus: Stimulus, parameters: pd.DataFrame) -> Tensor:
        """
        Predict a simple population receptive field model response to a stimulus.

        Parameters
        ----------
        stimulus : Stimulus
            Stimulus object.
        parameters : pandas.DataFrame
            Dataframe with columns containing different (sub-) model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        Tensor
            Model predictions of shape (num_voxels, num_frames). The number of voxels is the number of rows in
            `parameters`. The number of frames is the number of frames in the stimulus design.

        """
        response = self.models.prf_model(stimulus, parameters)
        design = ops.convert_to_tensor(stimulus.design)
        response = encode_prf_response(response, design)

        if self.models.impulse_model is not None:
            impulse_response = self.models.impulse_model(parameters)
            response = convolve_prf_impulse_response(response, impulse_response)

        if self.models.linear_model is not None:
            response = self.models.linear_model(response, parameters)

        return response
