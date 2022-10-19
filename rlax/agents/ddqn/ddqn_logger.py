import wandb
from observable import Observable
from codetiming import Timer
import jax
from jax.flatten_util import ravel_pytree


class DoubleQLogger:
    def __init__(self, mode, group, wandb_cfg) -> None:

        mode = mode
        group = group
        wandb.init(
            project="Rl-benchmark",
            config=wandb_cfg,
            entity="oserris",
            mode=mode,
            group=group,
        )

    def register(self, observable: Observable):
        observable.on("action_selection", self.action_selection)
        observable.on("on_learn_step", self.on_learn_step)
        observable.on("on_evaluation", self.on_evaluation)

    def action_selection(self, step, actions = None, epsilon=None, **kwargs):
        if epsilon:
            wandb.log({"explore/epsilon": epsilon}, step=step)

    def on_learn_step(self, step, loss, grads, **kwargs):
        flattened_grads, _ = ravel_pytree(grads)
        grad_norm = jax.numpy.linalg.norm(flattened_grads)
        wandb.log(
            {"learn/Q_loss": loss, "learn/Q_grad": grad_norm},
            step=step,
        )

    def on_evaluation(self, step, crewards, **kwargs):
        total = sum(Timer.timers.total(name) for name in Timer.timers.data)
        timers = {
            f"timers/{name}": Timer.timers.total(name) for name in Timer.timers.data
        }
        wandb.log({f"eval/creward{i}": cr for i, cr in enumerate(crewards)}, step=step)
        wandb.log({f"eval/mean_creward": crewards.mean()}, step=step)
        wandb.log(timers, step=step)
        wandb.log({"timers/total": total}, step=step)
