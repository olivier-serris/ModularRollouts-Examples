from gym import spaces
import jax
import jax.numpy as jnp
import collections
from brax.training.replay_buffers import UniformSamplingQueue
from jax.flatten_util import ravel_pytree


def get_uniform_action_sample_fct(single_actions_pace, action_space):
    def sample_discrete(key):
        return jax.random.choice(
            key,
            jnp.arange(single_actions_pace.n),
            shape=action_space.shape,
        )

    def sample_continuous(key):
        act = jax.random.uniform(key, shape=action_space.shape)
        act = (act - 0.5) * 2
        return act

    if isinstance(single_actions_pace, spaces.Discrete):
        return sample_discrete
    elif isinstance(single_actions_pace, spaces.Box):
        return sample_continuous
    else:
        raise NotImplementedError()


def check_env(env):
    if isinstance(env.action_space, spaces.Box):
        assert env.single_action_space.low == -1
        assert env.single_action_space.high == 1


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def grad_norm(grad):
    flattened_grads, _ = ravel_pytree(grad)
    return jax.numpy.linalg.norm(flattened_grads)


class ReplayBuffer(UniformSamplingQueue):
    def sample_with_key(self, buffer_state, key):
        _, sample = self.sample(buffer_state.replace(key=key))
        return sample
