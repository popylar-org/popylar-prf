"""Documentation about popylar_prf."""

import logging
from popylar_prf.stimulus import Stimulus
from popylar_prf.typing import Tensor
from popylar_prf.utils import convert_parameters_to_tensor

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.0.1"


__all__ = [
    "Stimulus",
    "Tensor",
    "convert_parameters_to_tensor",
]
