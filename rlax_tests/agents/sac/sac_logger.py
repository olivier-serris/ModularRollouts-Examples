import wandb
from observable import Observable
import jax
from jax.flatten_util import ravel_pytree
from logger import WandbLogger


class SACLogger(WandbLogger):
    def register(self, observable: Observable):
        super().register(observable)
        observable.on("on_learn_step", self.on_learn_step)
        observable.on("on_action_selection", self.on_action_selection)

    def on_action_selection(self, step, std=[], actions=[], **kwargs):
        if len(std):
            wandb.log(
                {
                    "learn/mean_std": std.mean(),
                },
                step=step,
            )
        if len(actions):
            wandb.log({"learn/mean_action": actions.mean()}, step=step)

    def on_learn_step(
        self,
        step,
        q_losses,
        actor_loss,
        q_grad_norms,
        actor_grad_norms,
        entropy_coeff,
        **kwargs
    ):
        wandb.log(
            {
                "learn/q_loss": q_losses.mean(),
                "learn/q_grad": q_grad_norms.mean(),
                "learn/actor_loss": actor_loss.mean(),
                "learn/actor_grad": actor_grad_norms.mean(),
                "learn/entropy_coeff": entropy_coeff.mean(),
            },
            step=step,
        )
