"""Backend-specific fitter base classes."""

from abc import abstractmethod
from collections.abc import Sequence
from typing import TypeAlias
import keras
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor

SGDState: TypeAlias = tuple[list, list, list, list] | None
"""State with objects that are carried during the stochastic gradient descent optimization."""


class ParamsDict:
    """
    A dictionary-like object that supports dataframe-style column selection but returns Keras tensors.

    Serves as an adapter during fitting to supply parameters to models while avoiding converting tensors into
    actual dataframes.

    Parameters
    ----------
    data: dict
        Dictionary of parameter tensors to perform column style selection on.

    """

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, keys: Sequence[str]) -> Tensor:
        return keras.ops.transpose(keras.ops.convert_to_tensor([self._data[k] for k in keys]))


class BaseSGDFitter(keras.Model):
    """Backend-specific stochastic gradient descent base class."""

    @abstractmethod
    def _get_state(self) -> SGDState:
        pass

    @abstractmethod
    def _update_model_weights(self, x: Stimulus, y: Tensor, state: SGDState) -> tuple[dict, SGDState]:
        pass
