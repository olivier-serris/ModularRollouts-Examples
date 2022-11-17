from typing import Any, Dict, List, NamedTuple
import chex
import jax
import haiku as hk
from haiku import nets
import jax.numpy as jnp
import optax
from gym.spaces import Box
import distrax
import collections
from utils import grad_norm

Distribution = collections.namedtuple(
    "Distribution", "sample, prob,log_prob,sample_and_log_prob"
)


def safe_tanh(x):
    return jnp.clip(jnp.tanh(x), a_min=-1.0, a_max=1.0)


def multivariate_gaussian():
    def squashed(mean, std):
        bijector = distrax.Tanh()
        return distrax.Transformed(
            distribution=distrax.MultivariateNormalDiag(loc=mean, scale_diag=std),
            bijector=distrax.Block(bijector, ndims=1),
        )

    def sample(key, mean, std):
        return squashed(mean, std).sample(seed=key)

    def prob(sample, mean, std):
        return squashed(mean, std).prob(sample)

    def log_prob(sample, mean, std):
        return squashed(mean, std).log_prob(sample)

    def sample_and_log_prob(key, mean, std):
        return squashed(mean, std).sample_and_log_prob(seed=key)

    return Distribution(sample, prob, log_prob, sample_and_log_prob)


def build_actor_continuous(hidden_layers: List, n_outputs: int) -> hk.Transformed:
    """Factory for a simple MLP actor with squashed gaussian for continuous actions."""

    def pi(obs):
        trunc = hk.Sequential(
            [hk.Flatten(), nets.MLP([*hidden_layers], activate_final=True)]
        )
        mu_layer = hk.Linear(n_outputs)
        std_layer = hk.Linear(n_outputs)
        latent = trunc(obs)
        mu, std = mu_layer(latent), std_layer(latent)
        # some implementations apply a tanh to mu.
        squashed_mu = jnp.tanh(mu)

        # We need constraint to force the std to be positive.
        # softplus, the continuous version of relu have nice properties to enforce positivity.
        # But it's necessary to add a epsilon (1e-5) to ensure no Nans arise in the derivative.
        # discussion : https://github.com/tensorflow/probability/issues/751
        positive_std = jax.nn.softplus(std) + 1e-5

        return squashed_mu, positive_std

    return hk.without_apply_rng(hk.transform(pi))


def build_entropy_net(init_val):
    def forward():
        log_coeff = hk.get_parameter(
            "entropy_coeff", [], init=hk.initializers.Constant(jnp.log(init_val))
        )
        return jnp.exp(log_coeff)

    return hk.without_apply_rng(hk.transform(forward))


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

    entropy_params: Dict
    entropy_opt_state: Dict


# Output of the action selection step :
class ActorOutput(NamedTuple):
    actions: jnp.array
    mean: jnp.array
    std: jnp.array


# Output of a learn step :
class LearnOutput(NamedTuple):
    q_losses: jnp.array
    q_grad_norms: Any
    actor_loss: jnp.array
    actor_grad_norms: Any
    entropy_coeff: float


class ActorLossInfos(NamedTuple):
    loss: jnp.array
    log_probs: jnp.array


class SAC:
    def __init__(
        self,
        action_space: Box,
        discount,
        actor_learning_rate,
        critic_learning_rate,
        entropy_coeff,
        entropy_learning_rate,
        max_grad_norm,
        critic_polyak_update_val,
        critic_network,  # TODO : change name
        actor_network,  # TODO : change name
        target_entropy="auto",
        update_q_target_every=1,
        n_critics=2,
    ) -> None:
        self._actor_net = build_actor_continuous(
            **actor_network, n_outputs=action_space.shape[-1]
        )
        self._q_net = build_Q_continuous(**critic_network)
        self._actor_optimizer = self._optimizer = optax.chain(
            optax.adam(actor_learning_rate), optax.clip_by_global_norm(max_grad_norm)
        )
        self._critic_optimizer = optax.chain(
            optax.adam(critic_learning_rate), optax.clip_by_global_norm(max_grad_norm)
        )
        self._entropy_optimizer = optax.chain(
            optax.adam(entropy_learning_rate), optax.clip_by_global_norm(max_grad_norm)
        )

        self._discount = discount
        self.squashed_gaussian = multivariate_gaussian()
        self.critic_polyak_update_val = critic_polyak_update_val
        self.n_critics = n_critics
        assert update_q_target_every == 1, "not implemented"
        self._update_q_target_every = update_q_target_every

        self.entropy_net = build_entropy_net(entropy_coeff)
        if target_entropy == "auto":
            self.target_entropy = -action_space.shape[-1]
        else:
            self.target_entropy = target_entropy

        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)
        self.learn_n_step = jax.jit(self.learn_n_step)

    def initialize(self, dummy_obs, key):
        k1, k2, k3 = jax.random.split(key, 3)
        # init entropy_coeff :
        entropy_params = self.entropy_net.init(k1)
        entropy_opt_state = self._entropy_optimizer.init(entropy_params)

        # Init actor:
        actor_params = self._actor_net.init(k2, dummy_obs)
        actor_opt_state = self._actor_optimizer.init(actor_params)

        # init critic :
        keys = jax.random.split(k3, self.n_critics)
        dummy_act, _ = self._actor_net.apply(actor_params, dummy_obs)
        q_params = jax.vmap(self._q_net.init, in_axes=(0, None, None))(
            keys, dummy_obs, dummy_act
        )
        # TODO : i have multiple critic optimizers, i should be able to have only one.
        q_opt_states = jax.vmap(self._critic_optimizer.init, in_axes=(0))(q_params)

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
            entropy_params=entropy_params,
            entropy_opt_state=entropy_opt_state,
        )

    def sample_action(self, actor_params, obs, key):
        mean, std = self._actor_net.apply(actor_params, obs)
        sampled_actions, log_prob = self.squashed_gaussian.sample_and_log_prob(
            key, mean, std
        )
        eval_actions = safe_tanh(mean)
        # actions are clipped because of tanh, the output is sometimes ~= 1.00001
        sampled_actions = jnp.clip(sampled_actions, a_min=-1.0, a_max=1.0)
        chex.assert_rank([mean, std, sampled_actions, eval_actions], 2)
        return log_prob, std, sampled_actions, eval_actions

    def actor_step(self, agent_state: SAC_State, obs, key, evaluation):
        print("actor_step COMPILED")
        _, std, sampled_actions, eval_actions = self.sample_action(
            agent_state.actor_params, obs, key
        )
        actions = jax.lax.select(evaluation, eval_actions, sampled_actions)
        return (
            agent_state,
            ActorOutput(actions=actions, mean=eval_actions, std=std),
        )

    def _actor_loss(self, actor_params, q_params, entropy_params, key, obs_t):
        log_probs, _, a_t, _ = self.sample_action(actor_params, obs_t, key)
        q_ts = self._batched_q_net(q_params, obs_t, a_t).squeeze()
        q_t = jnp.min(q_ts, axis=0)
        entropy = -log_probs
        entropy = entropy.squeeze()
        entropy_coeff = self.entropy_net.apply(entropy_params)
        chex.assert_rank([obs_t, q_ts, q_t, entropy, entropy_coeff], [2, 2, 1, 1, 0])
        loss = -q_t.mean() - entropy_coeff * entropy.mean()
        return loss, ActorLossInfos(loss, log_probs)

    def _actor_grad_step(self, agent_state: SAC_State, key, obs_t):
        actor_grads, loss_infos = jax.grad(self._actor_loss, has_aux=True)(
            agent_state.actor_params,
            agent_state.q_params,
            agent_state.entropy_params,
            key,
            obs_t,
        )
        actor_loss, log_probs = loss_infos.loss, loss_infos.log_probs

        updates, actor_opt_state = self._actor_optimizer.update(
            actor_grads, agent_state.actor_opt_state
        )
        actor_params = optax.apply_updates(agent_state.actor_params, updates)
        return (
            actor_params,
            actor_opt_state,
            actor_loss,
            grad_norm(actor_grads),
            log_probs,
        )

    def _get_q_target_val(self, agent_state: SAC_State, key, r_t, discount_t, obs_t):
        chex.assert_rank([r_t, discount_t, obs_t], [1, 1, 2])
        log_prob, _, a_t, _ = self.sample_action(agent_state.actor_params, obs_t, key)
        q_ts = self._batched_q_net(agent_state.q_target_params, obs_t, a_t).squeeze()
        q_t = jnp.min(q_ts, axis=0)
        entropy = -log_prob
        entropy = entropy.squeeze()
        entropy_coeff = self.entropy_net.apply(agent_state.entropy_params)
        y = r_t + discount_t * (q_t + entropy_coeff * entropy)
        chex.assert_rank([q_ts, q_t, entropy, y], [2, 1, 1, 1])
        return y

    def _one_critic_loss(self, q_param, obs_tm1, a_tm1, target_q_val):
        q_tm1 = self._q_net.apply(q_param, obs_tm1, a_tm1).squeeze()
        chex.assert_rank([q_tm1, a_tm1, target_q_val], [1, 2, 1])
        return (0.5 * (q_tm1 - target_q_val) ** 2).mean()

    def _one_critic_grad_step(
        self, q_params, q_opt_state, obs_tm1, a_tm1, target_q_val
    ):
        q_loss, q_grad = jax.value_and_grad(self._one_critic_loss)(
            q_params, obs_tm1, a_tm1, target_q_val
        )
        updates, q_opt_state = self._critic_optimizer.update(q_grad, q_opt_state)
        q_params = optax.apply_updates(q_params, updates)
        return q_params, q_opt_state, q_loss, grad_norm(q_grad)

    def _entropy_grad_step(
        self, entropy_opt_state, entropy_params, log_probs, target_entropy
    ):
        def _entropy_loss(params):
            entropy_coeff = self.entropy_net.apply(params)
            log_prob = log_probs.mean()
            loss = -(entropy_coeff * (log_prob + target_entropy))
            return loss

        grads = jax.grad(_entropy_loss)(entropy_params)
        updates, entropy_opt_state = self._entropy_optimizer.update(
            grads, entropy_opt_state
        )
        entropy_params = optax.apply_updates(entropy_params, updates)
        return entropy_params, entropy_opt_state

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

        q_params, q_opt_states, q_losses, q_grad_norms = self._batched_critic_step(
            agent_state.q_params, agent_state.q_opt_states, obs_tm1, a_tm1, q_target_val
        )
        ##### ACTOR GRAD STEP #####
        (
            actor_params,
            actor_opt_state,
            actor_loss,
            actor_grad_norms,
            log_probs,
        ) = self._actor_grad_step(agent_state, key, obs_t)

        ##### Entropy GRAD STEP #####
        entropy_params, entropy_opt_state = self._entropy_grad_step(
            agent_state.entropy_opt_state,
            agent_state.entropy_params,
            log_probs,
            self.target_entropy,
        )

        return (
            agent_state._replace(
                q_params=q_params,
                q_target_params=q_target_params,
                q_opt_states=q_opt_states,
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                learn_step=agent_state.learn_step + 1,
                entropy_params=entropy_params,
                entropy_opt_state=entropy_opt_state,
            ),
            LearnOutput(
                q_losses=q_losses,
                q_grad_norms=q_grad_norms,
                actor_loss=actor_loss,
                actor_grad_norms=actor_grad_norms,
                entropy_coeff=self.entropy_net.apply(entropy_params),
            ),
        )

    def learn_n_step(self, agent_state: SAC_State, buffer_samples, key):
        print("learn_n_step COMPILED")
        keys = jax.random.split(key, buffer_samples[0].shape[0])

        def learner_one_step(state, data):
            buffer_sample, rng_key = data["sample"], data["key"]
            return self.learner_step(state, buffer_sample, rng_key)

        iterated = {"sample": buffer_samples, "key": keys}
        return jax.lax.scan(learner_one_step, init=agent_state, xs=iterated)
