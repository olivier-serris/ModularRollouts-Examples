import jax
import haiku as hk
from haiku import nets
import collections
import jax.numpy as jnp
import optax
import rlax


def build_Q_discrete(hidden_layers, n_actions: int) -> hk.Transformed:
    """Factory for a simple MLP network for approximating Q-values."""

    def q(obs):
        network = hk.Sequential([hk.Flatten(), nets.MLP([*hidden_layers, n_actions])])
        return network(obs)

    return hk.without_apply_rng(hk.transform(q))


Params = collections.namedtuple(
    "Params", "online target"
)  # correspond to online network and target networks.
ActorState = collections.namedtuple("ActorState", "count epsilon")
ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
LearnerState = collections.namedtuple("LearnerState", "count opt_state loss grads")
# Data = collections.namedtuple("Data", "obs_tm1 a_tm1 r_t discount_t obs_t")


class DoubleQAgent:
    def __init__(
        self,
        n_actions,
        discount,
        hidden_layers,
        learning_rate,
        target_period,
        epsilon_cfg,
    ) -> None:

        self._network = build_Q_discrete(
            hidden_layers=hidden_layers, n_actions=n_actions
        )
        self._optimizer = optax.adam(learning_rate)
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
        self._target_period = target_period
        self._discount = discount

        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)

    def initial_params(self, dummy_obs, key):
        online_params = self._network.init(key, dummy_obs)
        return Params(online_params, online_params)

    def initial_actor_state(self):
        actor_count = jnp.zeros((), dtype=jnp.float32)
        return ActorState(actor_count, None)

    def initial_learner_state(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        opt_state = self._optimizer.init(params.online)
        return LearnerState(learner_count, opt_state, None, None)

    def actor_step(self, params, obs, actor_state, key, evaluation):
        # generate q_values :
        q = self._network.apply(params.online, obs)

        # get the current epsilon value :
        epsilon = self._epsilon_by_frame(actor_state.count)

        # get train and eval actions :
        train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
        eval_a = rlax.greedy().sample(key, q)
        a = jax.lax.select(evaluation, eval_a, train_a)
        return ActorOutput(actions=a, q_values=q), ActorState(
            actor_state.count + 1, epsilon
        )

    def _loss(
        self, online_params, target_params, obs_tm1, a_tm1, r_t, discount_t, obs_t
    ):
        """Given current (online) and target params of the q network,
        and data (s,a,r,s'); compute the double q learning loss"""
        q_tm1 = self._network.apply(online_params, obs_tm1)  # Q(s,.)
        q_t_val = self._network.apply(target_params, obs_t)  # Q_target(s,.)
        q_t_select = self._network.apply(online_params, obs_t)  #
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, q_t_select)
        return jnp.mean(rlax.l2_loss(td_error))

    def learner_step(self, params, learner_state, buffer_sample):
        # Update parameters every _target_period steps.
        target_params = optax.periodic_update(
            params.online, params.target, learner_state.count, self._target_period
        )
        (last_obs, actions, reward, observation, terminated) = buffer_sample
        actions = actions.reshape(-1).astype(jnp.int32)
        reward = reward.reshape(-1)
        terminated = terminated.reshape(-1)
        discount = jnp.where(terminated == 1, 0, self._discount)
        batched_step = (last_obs, actions, reward, discount, observation)

        loss, dloss_dtheta = jax.value_and_grad(self._loss)(
            params.online, target_params, *batched_step
        )
        updates, opt_state = self._optimizer.update(
            dloss_dtheta, learner_state.opt_state
        )
        online_params = optax.apply_updates(params.online, updates)
        return (
            Params(online_params, target_params),
            LearnerState(
                count=learner_state.count + 1,
                opt_state=opt_state,
                loss=loss,
                grads=dloss_dtheta,
            ),
        )
