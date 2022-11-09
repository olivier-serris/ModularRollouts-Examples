import collections
import wandb
from observable import Observable
from abc import ABC, abstractmethod
from utils import flatten_dict
from codetiming import Timer


class Logger(ABC):
    @abstractmethod
    def register(self, observable):
        pass

    @abstractmethod
    def on_learn_step(self, **kwargs):
        pass

    @abstractmethod
    def on_evaluation(self, **kwargs):
        pass


class WandbLogger(Logger):
    def __init__(self, mode, group, wandb_cfg) -> None:
        mode = mode
        group = group
        wandb.init(
            project="RL-benchmark",
            config=flatten_dict(wandb_cfg, sep="/"),
            entity="oserris",
            mode=mode,
            group=group,
        )
        self.since_last_eval_timer = Timer("since_last_eval")
        self.since_last_eval_timer.start()

    def register(self, observable: Observable):
        observable.on("on_evaluation", self.on_evaluation)

    def on_evaluation(self, step, crewards, eval_every, **kwargs):
        wandb.log({f"eval/creward{i}": cr for i, cr in enumerate(crewards)}, step=step)
        wandb.log({f"eval/mean_creward": crewards.mean()}, step=step)

        timers = {
            f"timers/{name}": Timer.timers.total(name)
            for name in Timer.timers.data
            if name != "since_last_eval"
        }
        wandb.log(timers, step=step)

        total = sum(
            Timer.timers.total(name)
            for name in Timer.timers.data
            if name != "since_last_eval"
        )
        wandb.log({"timers/total": total}, step=step)

        timers_norm = {
            f"timers/{name}_norm": Timer.timers.total(name) / total
            for name in Timer.timers.data
        }
        wandb.log(timers_norm, step=step)

        elapsed = self.since_last_eval_timer.stop()
        wandb.log({"timers/step_per_sec": eval_every / elapsed}, step=step)
        self.since_last_eval_timer.start()

        wandb.log({"timers/step": step}, step=step)

        print(f"step : {step} \tevals : {crewards.mean()} \telapsed : {total}")
