"""Stochastic gradient descent fitters."""

from collections.abc import Callable
from collections.abc import Sequence
import keras
import numpy as np
import pandas as pd
from keras import ops
from tqdm import tqdm
from prfmodel.models.base import BaseModel
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor

match keras.backend.backend():
    case "jax":
        from .backend.jax import JAXSGDFitter as BackendSGDFitter
    case "tensorflow":
        from .backend.tensorflow import TensorFlowSGDFitter as BackendSGDFitter
    case "torch":
        from .backend.torch import TorchSGDFitter as BackendSGDFitter
    case other:
        msg = f"Backend '{other}' is not supported."
        raise ValueError(msg)


class SGDFitter(BackendSGDFitter):
    """
    Fit population receptive field models with stochastic gradient descent (SGD).

    Estimates model parameters iteratively through SGD by minimizing the loss between model predictions and target
    data.

    Parameters
    ----------
    model : BaseModel
        Population receptive field model instance that can be fit to data.
        The model must implement `__call__` to make predictions that can be compared to data.
    stimulus : Stimulus
        Stimulus object used to make model predictions.
    optimizer : keras.optimizers.Optimizer, optional
        Optimizer instance. Default is `None` where a `keras.optimizers.Adam` optimizer is used.
    loss : keras.optimizers.Loss or Callable, optional
        Loss instance or function with the signatur `f(y, y_pred)`, where `y` is the target data and `y_pred` are the
        model predicitons. Default is `None` where a `keras.optimizers.MeanSquaredError` loss is used.

    Notes
    -----
    At each step during the fitting, the `model` makes a prediction for each batch in the target data
    given the `stimulus` and the current parameter values. The predictions are then compared to the target data and
    the parameter values are updated given the `optimizer` schedule.

    """

    def __init__(
        self,
        model: BaseModel,
        stimulus: Stimulus,
        optimizer: keras.optimizers.Optimizer | None = None,
        loss: keras.losses.Loss | Callable | None = None,
    ):
        super().__init__()

        self.model = model
        self.stimulus = stimulus

        if optimizer is None:
            optimizer = keras.optimizers.Adam()

        if loss is None:
            loss = keras.losses.MeanSquaredError()

        self.optimizer = optimizer
        self.loss = loss

    def _create_variables(self, init_parameters: pd.DataFrame, fixed_parameters: Sequence[str]) -> None:
        for key, val in init_parameters.items():
            # TODO: Infer dtype from model/stimulus
            setattr(self, key, keras.Variable(val, dtype="float64", name=key, trainable=key not in fixed_parameters))

    def _delete_variables(self, init_parameters: pd.DataFrame) -> None:
        for key in init_parameters:
            delattr(self, key)

    def fit(
        self,
        data: Tensor | np.ndarray,
        init_parameters: pd.DataFrame,
        fixed_parameters: Sequence[str] | None = None,
        num_steps: int = 1000,
    ) -> tuple[dict, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : Tensor or numpy.ndarray
            Target data to fit the model to. Must have shape (num_batches, num_frames), where `num_batches` is the
            number of batches for which parameters are estimated and `num_frames` is the number of time steps.
        init_parameters : pandas.DataFrame
            Dataframe initial model parameters. Columns must contain different model parameters and
            rows parameter values for each batch in `data`.
        fixed_parameters : Sequence of str, optional
            Names of model parameters that are fixed to their starting values, i.e., their values are not optimized
            during the fitting.
        num_steps : int, default=1000
            Number of optimization steps.

        Returns
        -------
        dict
            A dictionary with the final loss value (TODO: and potentially other metrics).
        pandas.DataFrame
            A dataframe with final model parameters.

        """
        if fixed_parameters is None:
            fixed_parameters = []

        self._create_variables(init_parameters, fixed_parameters)

        self.optimizer.build(self.trainable_variables)

        self.compile(optimizer=self.optimizer, loss=self.loss)

        state = self._get_state()

        with tqdm(range(num_steps)) as pbar:
            for _ in pbar:
                logs, state = self._update_model_weights(self.stimulus, data, state)

                if logs:
                    display_logs = {}
                    for key, value in logs.items():
                        display_logs[key] = float(value)

                    pbar.set_postfix(display_logs)

        if state is not None:
            trainable_variables, non_trainable_variables, _, _ = state
            for variable, value in zip(self.trainable_variables, trainable_variables, strict=False):
                variable.assign(value)
            for variable, value in zip(self.non_trainable_variables, non_trainable_variables, strict=False):
                variable.assign(value)

        params = pd.DataFrame(
            {v.name: ops.convert_to_numpy(v.value) for v in self.trainable_variables + self.non_trainable_variables},
        )

        self._delete_variables(init_parameters)

        # Sort result param columns according to inititial parameter columns
        return logs, params[init_parameters.columns]
