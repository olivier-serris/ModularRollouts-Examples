import os
from codetiming import Timer
import jax
import jax.numpy as jnp
import chex
import haiku as hk
from observable import Observable
from omegaconf import OmegaConf
import hydra
from brax.training.replay_buffers import UniformSamplingQueue
from modular_rollouts import create_env
from evaluation import eval_rollouts
from agents.agent import AgentOffPolicy

# TODO : general reorganisation :
# => More efficent main train function for pure jax ?
# => If the replay buffer is moved inside the agent, it simplifies the genericity of the main loop
# add a collect n step module ?


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="ddqn_gym.yaml")
def train(hydra_config):
    ################ INIT ################
    cfg = OmegaConf.to_container(hydra_config, resolve=True)

    seed = cfg["seed"]
    num_envs = cfg["env"]["num_envs"]
    num_evals = cfg["env"]["num_eval"]
    eval_every = cfg["env"]["eval_every"]
    agent_cfg = cfg["train"]
    total_train_step = agent_cfg["total_train_step"]
    start_training_after_x_steps = agent_cfg["start_training_after_x_steps"]
    max_replay_size = agent_cfg["max_replay_size"]
    batch_size = agent_cfg["batch_size"]
    grad_steps_per_step = agent_cfg["grad_steps_per_step"]

    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

    # init env :
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

    # init agent :
    agent: AgentOffPolicy = hydra.utils.instantiate(
        cfg["train"]["agent"], action_space=env.single_action_space
    )
    agent_state = agent.initialize(dummy_obs=observation, key=next(rng))

    # init logger :
    event = Observable()
    logger = hydra.utils.instantiate(
        cfg["log"]["logger"], _recursive_=False, wandb_cfg=cfg
    )
    logger.register(event)

    # init replay buffer :
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
                actions = jax.random.choice(
                    next(rng),
                    jnp.arange(env.single_action_space.n),
                    shape=env.action_space.shape,
                )
                actor_output = {}
            else:
                agent_state, actor_output = agent.actor_step(
                    agent_state=agent_state,
                    obs=observation,
                    key=next(rng),
                    evaluation=False,
                )
                actions = actor_output.actions
                actor_output = actor_output._asdict()
        event.trigger(
            "action_selection",
            step=n_step_done,
            **actor_output,
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
                # TODO : optimize this, sample every sample in one go and then scan
                for _ in range(grad_steps_per_step):
                    buffer_state, buffer_sample = replaybuffer.sample(buffer_state)
                    agent_state, learner_output = agent.learner_step(
                        agent_state, buffer_sample
                    )
            event.trigger("on_learn_step", step=n_step_done, **learner_output._asdict())

        with Timer("evaluation", logger=None):
            if n_step_done % eval_every == 0:
                crewards = eval_rollouts(eval_env, agent, agent_state, rng)
        event.trigger(
            "on_evaluation",
            step=n_step_done,
            crewards=crewards,
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
