from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import jax
import jax.numpy as jnp
import keras
import numpy as np
import tensorflow as tf
import torch


def to_numpy_safe(x: tf.Tensor) -> np.ndarray:
    """A type-safe, multi-backend function to get a NumPy array."""
    backend = keras.backend.backend()

    if backend == "tensorflow":
        if TYPE_CHECKING:
            x = cast("tf.Tensor", x)
        return x.numpy()

    if backend == "torch":
        return x.numpy()

    if backend == "jax":
        return np.asarray(x)

    raise ValueError


class TorchModel(keras.Model):
    """Torch class."""

    def update_model_weights(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        state: str | None = None,
    ) -> tuple[dict[str, float], Any]:
        """Update model weights."""
        self.zero_grad()

        y_pred = self.model.predict(x, self.adapter.inverse({v.name: v.value for v in self.trainable_variables}))
        loss = self.compute_loss(y=y, y_pred=y_pred)

        loss.backward()

        trainable_weights = list(self.trainable_weights)
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}

        return logs, state


class JAXModel(keras.Model):
    """Jax model."""

    def compute_loss_and_updates(
        self,
        trainable_variables: list[jax.Array],
        non_trainable_variables: list[jax.Array],
        x: tf.Tensor,
        y: tf.Tensor,
    ) -> tuple[
        jax.Array,
        tuple[list[jax.Array], list[jax.Array]],
    ]:
        """Update model weights."""
        # Convert JAX arrays to ensure compatibility
        trainable_variables = [jnp.array(v) for v in trainable_variables]
        non_trainable_variables = [jnp.array(v) for v in non_trainable_variables]

        state_mapping: list[tuple[jax.Array, jax.Array]] = []
        state_mapping.extend(zip(self.trainable_variables, trainable_variables, strict=False))
        state_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables, strict=False))

        with keras.StatelessScope(state_mapping) as scope:
            # TODO: This is a bit hacky
            y_pred = self.model.predict(
                x,
                self.adapter.inverse(
                    {key.name: val for key, val in zip(self.trainable_variables, trainable_variables, strict=False)},
                ),
            )
            loss: jax.Array = self.compute_loss(y=y, y_pred=y_pred)  # TODO: why is this necessary?

        # update variables
        non_trainable_variables = [scope.get_current_value(v) for v in self.non_trainable_variables]

        return loss, (y_pred, non_trainable_variables)

    # TODO: here we need to define the types in the class definition? what does `super` define them as?
    def get_state(self) -> tuple[tf.Tensor]:
        """Get state."""
        return self.trainable_variables, self.non_trainable_variables, self.optimizer.variables, self.metrics_variables

    def update_model_weights(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        state: list[tf.Tensor],
    ) -> tuple[dict[str, float], list[tf.Tensor]]:
        """Update weights."""
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state

        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            x,
            y,
        )

        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables,
            grads,
            trainable_variables,
        )

        new_metrics_vars: list[tf.Tensor] = []
        logs: dict[str, float] = {}
        for metric in self.metrics:
            this_metric_vars = metrics_variables[len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars,
                    y,
                    y_pred,
                )
            logs[metric.name] = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        state = [
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,  # this is keeps creating problems
        ]

        return logs, state
