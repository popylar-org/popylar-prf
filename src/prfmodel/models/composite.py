"""Composite population receptive field models."""

from typing import cast
import pandas as pd
from keras import ops
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from .base import BaseImpulse
from .base import BasePRFModel
from .base import BasePRFResponse
from .base import BaseTemporal
from .encoding import encode_prf_response
from .impulse import TwoGammaImpulse
from .impulse import convolve_prf_impulse_response
from .temporal import BaselineAmplitude

DefaultImpulse = TwoGammaImpulse()
DefaultTemporal = BaselineAmplitude()


class SimplePRFModel(BasePRFModel):
    """
    Simple composite population receptive field model.

    This is a generic class that combines a population receptive field, impulse, and temporal response.

    Parameters
    ----------
    prf_model : BasePRFResponse
        A population receptive field response model instance.
    impulse_model : BaseImpulse or None, default=DefaultImpulse, optional
        An impulse response model instance. The default is a `TwoGammaImpulse` instance with default
        values.
    temporal_model : BaseTemporal or None, default=DefaultTemporal, optional
        An temporal model instance. The default is a `BaselineAmplitude` instance.

    Notes
    -----
    The simple composite model follows five steps:

    1. The population receptive field response model makes a prediction for the stimulus grid.
    2. The response is encoded with the stimulus design.
    3. A impulse response model generates an impulse response.
    4. The encoded response is convolved with the impulse response.
    5. The temporal model modifies the convolved response.

    """

    def __init__(
        self,
        prf_model: BasePRFResponse,
        impulse_model: BaseImpulse | None = DefaultImpulse,
        temporal_model: BaseTemporal | None = DefaultTemporal,
    ):
        super().__init__(
            prf_model=prf_model,
            impulse_model=impulse_model,
            temporal_model=temporal_model,
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
        prf_model = cast("BasePRFResponse", self.models["prf_model"])
        response = prf_model(stimulus, parameters)  # type: ignore[misc]
        design = ops.convert_to_tensor(stimulus.design)
        response = encode_prf_response(response, design)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters)
            response = convolve_prf_impulse_response(response, impulse_response)

        if self.models["temporal_model"] is not None:
            temporal_model = cast("BaseTemporal", self.models["temporal_model"])
            response = temporal_model(response, parameters)

        return response
