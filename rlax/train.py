from codetiming import Timer
import os
import jax

import haiku as hk
import jax.numpy as jnp
from observable import Observable
from omegaconf import OmegaConf
from brax.training.replay_buffers import UniformSamplingQueue
import hydra
from ddqn.ddqn import DoubleQAgent
from ddqn.ddqn_logger import ddqn_logger
from evaluation import eval_rollouts
from modular_rollouts import create_env
import chex


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="ddqn_gym.yaml")
def train(cfg):
    ################ INIT ################
    cfg = OmegaConf.to_container(cfg, resolve=True)
    agent_cfg = cfg["train"]
    seed = cfg["seed"]
    total_train_step = agent_cfg["total_train_step"]
    start_training_after_x_steps = agent_cfg["start_training_after_x_steps"]
    max_replay_size = agent_cfg["max_replay_size"]
    batch_size = agent_cfg["batch_size"]
    num_envs = cfg["env"]["num_envs"]
    num_evals = cfg["env"]["num_eval"]
    target_period = agent_cfg["target_period"]
    discount = agent_cfg["discount"]
    eval_every = cfg["env"]["eval_every"]

    event = Observable()
    logger = ddqn_logger(cfg)
    logger.register(event)
    env, eval_env = create_env(
        env_engine=cfg["env"]["env_engine"],
        env_name=cfg["env"]["name"],
        n_eval_env=num_evals,
        n_train_env=num_envs,
        action_type=chex.Array,
        n_pop=1,
        seed=seed,
    )
    observation, info = env.reset(seed=seed)
    agent = DoubleQAgent(
        env.single_action_space.n,
        discount=discount,
        hidden_layers=agent_cfg["hidden_layers"],
        learning_rate=agent_cfg["learning_rate"],
        target_period=target_period,
        epsilon_cfg=dict(
            init_value=1,
            end_value=0.01,
            transition_steps=1000,
            power=1.0,
        ),
    )
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    params = agent.initial_params(dummy_obs=observation, key=next(rng))
    learner_state = agent.initial_learner_state(params)
    actor_state = agent.initial_actor_state()
    action_shape = (
        1 if env.single_action_space.shape == () else env.single_action_space.shape
    )
    dummy_step = (
        jnp.zeros(env.single_observation_space.shape),  # obs_{t-1}
        jnp.zeros(action_shape),  # action_{t-1}
        jnp.zeros(1),  # reward_t
        jnp.zeros(env.single_observation_space.shape),  # obs_t
        jnp.zeros(1),  # terminated_t
    )

    replaybuffer = UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_step,
        sample_batch_size=batch_size,
    )
    buffer_state = replaybuffer.init(next(rng))
    replaybuffer.sample = jax.jit(replaybuffer.sample)
    replaybuffer.insert = jax.jit(replaybuffer.insert)

    ################ TRAIN ################

    n_step_done = 0
    last_obs = jnp.array(observation)
    last_done = jnp.zeros(env.num_envs, dtype=bool)
    while n_step_done <= total_train_step:
        with Timer(name="action_selection", logger=None):
            if n_step_done < start_training_after_x_steps:
                actions = jax.random.uniform(next(rng), env.single_action_space.shape)
            else:
                actor_output, actor_state = agent.actor_step(
                    params=params,
                    obs=observation,
                    actor_state=actor_state,
                    key=next(rng),
                    evaluation=False,
                )
                actions = actor_output.actions
        event.trigger(
            "action_selection",
            step=n_step_done,
            action=actions,
            **actor_state._asdict(),
        )
        with Timer("env_step", logger=None):
            env_output = env.step(actions)
        with Timer("replaybuffer_insert", logger=None):
            observation, reward, terminated, truncated, info = env_output
            done = jnp.logical_or(terminated, truncated)
            step_data = (last_obs, actions, reward, observation, terminated)
            # remove wrong steps (terminal -> start)
            step_data = jax.tree_util.tree_map(lambda x: x[~last_done], step_data)
            buffer_state = replaybuffer.insert(buffer_state, step_data)
        if (
            n_step_done > start_training_after_x_steps
            and replaybuffer.size(buffer_state) > start_training_after_x_steps
        ):
            with Timer("learn_step", logger=None):
                # training :
                buffer_state, buffer_sample = replaybuffer.sample(buffer_state)
                params, learner_state = agent.learner_step(
                    params, learner_state=learner_state, buffer_sample=buffer_sample
                )
            event.trigger("on_learn_step", step=n_step_done, **learner_state._asdict())

        with Timer("evaluation", logger=None):
            if n_step_done % eval_every == 0:
                crewards = eval_rollouts(eval_env, agent, actor_state, params, rng)
        event.trigger(
            "on_evaluation",
            step=n_step_done,
            crewards=crewards,
            **learner_state._asdict(),
        )
        last_obs = observation
        last_done = done
        n_step_done += env.num_envs

    env.close()
    total = sum(Timer.timers.total(name) for name in Timer.timers.data)
    print("total timers : ", total)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    train()
