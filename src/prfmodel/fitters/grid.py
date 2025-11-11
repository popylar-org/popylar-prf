"""Grid fitters."""

from collections.abc import Callable
from itertools import product
import keras
import numpy as np
import pandas as pd
from keras import ops
from more_itertools import chunked
from tqdm import tqdm
from prfmodel.models.base import BaseModel
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from .backend.base import ParamsDict


class GridHistory:
    """Grid search metric history.

    Logs losses and metrics over data batches resulting from a grid search.

    Attributes
    ----------
    history : dict
        Dictionary with keys indicating metric names and values containing metric values for each data batch.

    """

    def __init__(self, history: dict | None):
        self.history = history


class GridFitter:
    """Fit population receptive field models with grid search.

    Estimates model parameters by evaluating the model on a grid of parameter combinations and finding the
    minimum loss.

    Parameters
    ----------
    model : BaseModel
        Population receptive field model instance that can be fit to data.
        The model must implement `__call__` to make predictions that can be compared to data.
    stimulus : Stimulus
        Stimulus object used to make model predictions.
    loss : keras.optimizers.Loss or Callable, optional
        Loss instance or function with the signatur `f(y, y_pred)`, where `y` is the target data and `y_pred` are the
        model predicitons. Default is `None` where a `keras.optimizers.MeanSquaredError` loss is used. Note that, when
        a `keras.losses.Loss` instance is used, the argument `reduction` must be set to `'none'` to enable loss
        computation for all data batches.
    dtype : str, optional
        The dtype used for fitting. If `None` (the default), uses `keras.config.floatx()` which defaults
        to `float32`.

    Notes
    -----
    Depending on the size of the parameter grid and the number of batches in the data, the search can be very
    memory-intensive. For this reason, the grid is first split into chunks that are evaluated iteratively.

    """

    def __init__(
        self,
        model: BaseModel,
        stimulus: Stimulus,
        loss: keras.losses.Loss | Callable | None = None,
        dtype: str | None = None,
    ):
        self.model = model
        self.stimulus = stimulus

        if loss is None:
            loss = keras.losses.MeanSquaredError(reduction="none")

        self.loss = loss
        self.dtype = dtype

    def fit(
        self,
        data: Tensor,
        parameter_values: dict[str, Tensor | np.ndarray],
        chunk_size: int | None = None,
    ) -> tuple[GridHistory, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : Tensor
            Target data to fit the model to. Must have shape (num_batches, num_frames), where `num_batches` is the
            number of batches for which parameters are estimated and `num_frames` is the number of time steps.
        parameter_values : dict
            Dictionary with keys indicating model parameters and values indicating parameter values in the grid. The
            grid is constructed by taking all combinations of parameters values (i.e., the cartesian product).
        chunk_size : int, optional
            Size of each chunk of the grid that is evaluated at the same time.

        Returns
        -------
        GridHistory
            A history object that contains loss and metric values for each data batch.
        pandas.DataFrame
            A dataframe with final model parameters.

        """
        # Convert data to tensor
        data = ops.convert_to_tensor(data, dtype=self.dtype)
        # Keep parameter combinations as numpy arrays
        arrays = [ops.convert_to_numpy(val) for val in parameter_values.values()]
        # Calc total number of combinations
        total_grid_size = int(np.prod([len(x) for x in arrays]))

        if chunk_size is None:
            chunk_size = total_grid_size

        # Create generator for chunked parameter combinations
        param_iter_batched = chunked(product(*arrays), n=chunk_size)
        # Initialize array of best parameter set for each source
        best_params = ops.convert_to_numpy(ops.zeros((len(parameter_values.keys()), data.shape[0]), dtype=self.dtype))
        # Initialize array of best loss for each source
        best_loss = ops.convert_to_numpy(ops.full((data.shape[0],), fill_value=np.inf, dtype=self.dtype))
        # Add parameter combination dimension to data for broadcasting
        data = ops.expand_dims(data, 0)

        # Create progress bar
        with tqdm(
            param_iter_batched,
            desc="Processing parameter grid chunks",
            total=int(total_grid_size / chunk_size),
        ) as pbar:
            # Use for-loop to save memory
            for batch in pbar:
                # Stack tuples returned from generator
                params = np.stack(batch).T
                # Create a dict that allows column selection like data frames
                param_dict = ParamsDict(dict(zip(parameter_values.keys(), params, strict=False)))
                # Create model predictions for each parameter combination and add dimension for data sources
                pred = ops.expand_dims(self.model(self.stimulus, param_dict, dtype=self.dtype), 1)  # type: ignore[operator]
                # Calculate loss for each parameter combination and data source
                losses = self.loss(data, pred)
                # For each data source if min loss across all parameter combinations is lower, update
                min_loss = ops.convert_to_numpy(ops.amin(losses, axis=0))
                min_loss_idx = ops.convert_to_numpy(ops.argmin(losses, axis=0))
                loss_is_lower = min_loss < best_loss
                best_loss[loss_is_lower] = min_loss[loss_is_lower]
                best_params[:, loss_is_lower] = params[:, min_loss_idx[loss_is_lower]]
                # Add mean loss across data batches to progress bar
                pbar.set_postfix({"loss": float(best_loss.mean())})

        best_params_out = pd.DataFrame(dict(zip(parameter_values.keys(), best_params, strict=False)))

        return GridHistory({"loss": best_loss}), best_params_out
