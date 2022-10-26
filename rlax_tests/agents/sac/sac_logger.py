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

    def on_learn_step(self, step, q_losses, actor_loss, q_grads, actor_grads, **kwargs):
        flattened_grads, _ = ravel_pytree(q_grads)
        critic_grad_norm = jax.numpy.linalg.norm(flattened_grads)
        flattened_grads, _ = ravel_pytree(actor_grads)
        actor_grad_norm = jax.numpy.linalg.norm(flattened_grads)
        wandb.log(
            {
                "learn/q_loss": q_losses.mean(),
                "learn/q_grad": critic_grad_norm,
                "learn/actor_loss": actor_loss,
                "learn/actor_grad": actor_grad_norm,
            },
            step=step,
        )
