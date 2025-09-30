"""Documentation about popylar_prf."""

import logging
from popylar_prf.stimulus import Stimulus

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.0.1"


__all__ = [
    "Stimulus",
]
