import wandb
from observable import Observable
from codetiming import Timer
import jax
from jax.flatten_util import ravel_pytree
from logger import WandbLogger


class DoubleQLogger(WandbLogger):
    def register(self, observable: Observable):
        super().register(observable)
        observable.on("on_action_selection", self.on_action_selection)
        observable.on("on_learn_step", self.on_learn_step)

    def on_action_selection(self, step, actions=None, epsilon=None, **kwargs):
        if epsilon:
            wandb.log({"explore/epsilon": epsilon}, step=step)

    def on_learn_step(self, step, loss, grads, **kwargs):
        flattened_grads, _ = ravel_pytree(grads)
        grad_norm = jax.numpy.linalg.norm(flattened_grads)
        wandb.log(
            {"learn/q_loss": loss, "learn/q_grad": grad_norm},
            step=step,
        )
