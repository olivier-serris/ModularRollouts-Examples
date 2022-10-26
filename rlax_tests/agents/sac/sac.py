from typing import Any, Dict, List, NamedTuple
import chex
import jax
import haiku as hk
from haiku import nets
import jax.numpy as jnp
import optax
from gym.spaces import Box
from rlax import squashed_gaussian
import distrax
import collections

Distribution = collections.namedtuple("Distribution", "sample prob")


def multivariate_gaussian():
    def squashed(mean, std):
        bijector = distrax.Tanh()
        return distrax.Transformed(
            distribution=distrax.MultivariateNormalDiag(loc=mean, scale_diag=std),
            bijector=distrax.Block(bijector, ndims=1),
        )

    def sample(mean, std, key):
        squashed(mean, std).sample(key)

    def prob(mean, std):
        squashed(mean, std).prob()

    return Distribution(sample, prob)


def build_actor_continuous(hidden_layers: List, n_outputs: int) -> hk.Transformed:
    """Factory for a simple MLP actor with squashed gaussian for continuous actions."""
    # TODO : Verify what is the original SAC network ?
    # Is ther a better way to implement multi-head haiku networks ?
    def pi(obs):
        trunc = hk.Sequential([hk.Flatten(), nets.MLP([*hidden_layers])])
        mu_layer = hk.Linear(n_outputs)
        log_std_layer = hk.Linear(n_outputs)
        latent = trunc(obs)
        mu, log_std = mu_layer(latent), log_std_layer(latent)
        return mu, log_std

    return hk.without_apply_rng(hk.transform(pi))


def build_Q_continuous(hidden_layers: List) -> hk.Transformed:
    """Factory for a simple MLP network for approximating Q-values."""

    def q(obs, actions):
        network = hk.Sequential(
            [
                hk.Flatten(),
                nets.MLP([*hidden_layers, 1]),
            ]
        )
        input = jnp.concatenate([obs, actions], axis=1)
        return network(input)

    return hk.without_apply_rng(hk.transform(q))


# State of SAC algorithm :
class SAC_State(NamedTuple):
    learn_step: int
    q_params: Dict
    q_target_params: Dict
    q_opt_states: Dict
    actor_params: Dict
    actor_opt_state: Dict


# Output of the action selection step :
class ActorOutput(NamedTuple):
    actions: jnp.array
    mean: jnp.array
    std: jnp.array


# Output of a learn step :
class LearnOutput(NamedTuple):
    q_losses: jnp.array
    q_grads: Any
    actor_loss: jnp.array
    actor_grads: Any


class SAC:
    def __init__(
        self,
        action_space: Box,
        discount,
        learning_rate,
        entropy_coeff,
        max_grad_norm,
        critic_polyak_update_val,
        critic_network,
        actor_network,
        n_critics=2,
    ) -> None:
        self._actor_net = build_actor_continuous(
            **actor_network, n_outputs=action_space.shape[-1]
        )
        self._q_net = build_Q_continuous(**critic_network)
        self._optimizer = optax.chain(
            optax.adam(learning_rate), optax.clip_by_global_norm(max_grad_norm)
        )
        self._discount = discount
        self.squashed_gaussian = squashed_gaussian()
        self.entropy_coeff = entropy_coeff
        self.critic_polyak_update_val = critic_polyak_update_val
        self.n_critics = n_critics
        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)

    def initialize(self, dummy_obs, rng):
        actor_params = self._actor_net.init(next(rng), dummy_obs)
        dummy_act, _ = self._actor_net.apply(actor_params, dummy_obs)
        actor_opt_state = self._optimizer.init(actor_params)

        keys = jnp.array([next(rng) for _ in range(self.n_critics)])
        q_params = jax.vmap(self._q_net.init, in_axes=(0, None, None))(
            keys, dummy_obs, dummy_act
        )
        q_opt_states = jax.vmap(self._optimizer.init, in_axes=(0))(q_params)

        # creates iterate-over-critics batched functions :
        self._batched_q_net = jax.vmap(self._q_net.apply, in_axes=(0, None, None))
        self._batched_polyak = jax.vmap(optax.incremental_update, in_axes=(0, 0, None))
        self._batched_critic_step = jax.vmap(
            self._one_critic_grad_step, in_axes=(0, 0, None, None, None)
        )

        return SAC_State(
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            q_params=q_params,
            q_target_params=q_params,
            q_opt_states=q_opt_states,
            learn_step=jnp.zeros((), dtype=jnp.float32),
        )

    def sample_action(self, actor_params, obs, key):
        mean, log_std = self._actor_net.apply(actor_params, obs)
        actions = self.squashed_gaussian.sample(
            key, mean, log_std, action_spec=None
        )  # TODO : deprecated
        sampled_actions = actions
        eval_actions = jnp.tanh(mean)  # eval action like sampled actions need a tanh.
        chex.assert_rank([mean, log_std, sampled_actions, eval_actions], 2)
        return mean, log_std, sampled_actions, eval_actions

    def actor_step(self, agent_state: SAC_State, obs, key, evaluation):
        mean, log_std, sampled_actions, eval_actions = self.sample_action(
            agent_state.actor_params, obs, key
        )
        actions = jax.lax.select(evaluation, eval_actions, sampled_actions)
        return (
            agent_state,
            ActorOutput(actions=actions, mean=mean, std=jnp.exp(log_std)),
        )

    def _actor_loss(self, actor_params, q_params, key, obs_t):
        mean, log_std, a_t, _ = self.sample_action(actor_params, obs_t, key)
        q_ts = self._batched_q_net(q_params, obs_t, a_t).squeeze()
        q_t = jnp.min(q_ts, axis=0)
        entropy = -self.squashed_gaussian.logprob(a_t, mean, log_std, action_spec=None)
        entropy = entropy.squeeze()

        chex.assert_rank([obs_t, q_ts, q_t, entropy], [2, 2, 1, 1])

        return -q_t.mean() - self.entropy_coeff * entropy.mean()

    def _get_q_target_val(self, agent_state: SAC_State, key, r_t, discount_t, obs_t):
        chex.assert_rank([r_t, discount_t, obs_t], [1, 1, 2])
        mean, log_std, a_t, _ = self.sample_action(agent_state.actor_params, obs_t, key)
        q_ts = self._batched_q_net(agent_state.q_target_params, obs_t, a_t).squeeze()
        q_t = jnp.min(q_ts, axis=0)
        entropy = -self.squashed_gaussian.logprob(a_t, mean, log_std, action_spec=None)
        entropy = entropy.squeeze()
        y = r_t + discount_t * (q_t + self.entropy_coeff * entropy)
        chex.assert_rank([q_ts, q_t, entropy, y], [2, 1, 1, 1])
        return y

    def _one_critic_loss(self, q_param, obs_tm1, a_tm1, target_q_val):
        q_tm1 = self._q_net.apply(q_param, obs_tm1, a_tm1).squeeze()
        chex.assert_rank([q_tm1, a_tm1, target_q_val], [1, 2, 1])
        return ((q_tm1 - target_q_val) ** 2).mean()

    def _one_critic_grad_step(
        self, q_params, q_opt_state, obs_tm1, a_tm1, target_q_val
    ):
        q_loss, q_grad = jax.value_and_grad(self._one_critic_loss)(
            q_params, obs_tm1, a_tm1, target_q_val
        )
        updates, q_opt_state = self._optimizer.update(q_grad, q_opt_state)
        q_params = optax.apply_updates(q_params, updates)
        return q_params, q_opt_state, q_loss, q_grad

    def learner_step(self, agent_state: SAC_State, buffer_sample, key):
        # Update the q target networks by polyak averaging.
        q_target_params = self._batched_polyak(
            agent_state.q_params,
            agent_state.q_target_params,
            self.critic_polyak_update_val,
        )

        # construct data from the sampled batch data
        (obs_tm1, a_tm1, r_t, obs_t, terminated_t) = buffer_sample
        r_t = r_t.squeeze()
        terminated_t = terminated_t.squeeze()
        discount_t = jnp.where(terminated_t == 1, 0, self._discount)

        ##### CRITIC GRAD STEP #####
        q_target_val = self._get_q_target_val(agent_state, key, r_t, discount_t, obs_t)

        q_params, q_opt_states, q_losses, q_grads = self._batched_critic_step(
            agent_state.q_params, agent_state.q_opt_states, obs_tm1, a_tm1, q_target_val
        )
        ##### ACTOR GRAD STEP #####
        actor_loss, actor_grads = jax.value_and_grad(self._actor_loss)(
            agent_state.actor_params, q_params, key, obs_tm1
        )
        updates, actor_opt_state = self._optimizer.update(
            actor_grads, agent_state.actor_opt_state
        )
        actor_params = optax.apply_updates(agent_state.actor_params, updates)

        return (
            agent_state._replace(
                q_params=q_params,
                q_target_params=q_target_params,
                q_opt_states=q_opt_states,
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                learn_step=agent_state.learn_step + 1,
            ),
            LearnOutput(
                q_losses=q_losses,
                q_grads=q_grads,
                actor_loss=actor_loss,
                actor_grads=actor_grads,
            ),
        )

    def learn_n_step(self, agent_state: SAC_State, buffer_samples, keys):
        def learner_one_step(state, data):
            buffer_sample, key = data["sample"], data["key"]
            return self.learner_step(state, buffer_sample, key)

        iterated = {"sample": buffer_samples, "key": keys}
        return jax.lax.scan(learner_one_step, init=agent_state, xs=iterated)
