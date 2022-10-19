import jax
import haiku as hk
from haiku import nets
import collections
import jax.numpy as jnp
import optax
import rlax
from gym.spaces.discrete import Discrete
from wandb import agent


def build_Q_discrete(hidden_layers, n_actions: int) -> hk.Transformed:
    """Factory for a simple MLP network for approximating Q-values."""

    def q(obs):
        network = hk.Sequential([hk.Flatten(), nets.MLP([*hidden_layers, n_actions])])
        return network(obs)

    return hk.without_apply_rng(hk.transform(q))


# The state of the agent is decomposed into tree parts :
AgentState = collections.namedtuple("AgentState", "params actor_state learner_state")
Params = collections.namedtuple("Params", "online target")  # q_net and q_target
ActorState = collections.namedtuple("ActorState", "count epsilon")
LearnerState = collections.namedtuple("LearnerState", "count opt_state")

ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
LearnerOutput = collections.namedtuple("LearnerOutput", "loss grads")


class DoubleQAgent:
    def __init__(
        self,
        action_space: Discrete,
        discount,
        hidden_layers,
        learning_rate,
        target_period,
        epsilon_cfg,
    ) -> None:

        self._network = build_Q_discrete(
            hidden_layers=hidden_layers, n_actions=action_space.n
        )
        self._optimizer = optax.adam(learning_rate)
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
        self._target_period = target_period
        self._discount = discount

        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)

    def initialize(self, dummy_obs, key):
        online_params = self._network.init(key, dummy_obs)
        params = Params(online_params, online_params)

        actor_count = jnp.zeros((), dtype=jnp.float32)
        actor_state = ActorState(actor_count, None)

        learner_count = jnp.zeros((), dtype=jnp.float32)
        opt_state = self._optimizer.init(params.online)
        learner_state = LearnerState(learner_count, opt_state)

        return AgentState(
            params=params, actor_state=actor_state, learner_state=learner_state
        )

    def actor_step(self, agent_state: AgentState, obs, key, evaluation):
        actor_state = agent_state.actor_state
        params = agent_state.params
        # generate q_values :
        q = self._network.apply(params.online, obs)

        # get the current epsilon value :
        epsilon = self._epsilon_by_frame(actor_state.count)

        # get train and eval actions :
        train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
        eval_a = rlax.greedy().sample(key, q)
        actions = jax.lax.select(evaluation, eval_a, train_a)
        return (
            agent_state._replace(
                actor_state=ActorState(actor_state.count + 1, epsilon)
            ),
            ActorOutput(actions=actions, q_values=q),
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

    def learner_step(self, agent_state: AgentState, buffer_sample):
        (last_obs, actions, reward, observation, terminated) = buffer_sample
        params, learner_state = agent_state.params, agent_state.learner_state
        # Update parameters every _target_period steps.
        target_params = optax.periodic_update(
            params.online, params.target, learner_state.count, self._target_period
        )
        # TODO : do i still need all these reshapes ?
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
        new_agent_state = agent_state._replace(
            params=Params(online_params, target_params),
            learner_state=LearnerState(
                count=learner_state.count + 1,
                opt_state=opt_state,
            ),
        )
        return new_agent_state, LearnerOutput(loss=loss, grads=dloss_dtheta)
