import os, sys
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
import gym.spaces as spaces
from utils import check_env, get_uniform_action_sample_fct

# TODO : general reorganisation :
# => More efficent main train function for pure jax ?
# => If the replay buffer is moved inside the agent, it simplifies the genericity of the main loop
# add a collect n step module ?

# TODO : how to handle random keys in a clean way.

# sac_gym.yaml  ddqn_gym.yaml
@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="sac_gym.yaml")
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
        max_step=cfg["env"]["episode_max_length"],
        n_eval_env=num_evals,
        n_train_env=num_envs,
        action_type=chex.Array,
        n_pop=1,
        seed=seed,
    )
    # check_env(env)
    first_obs, info = env.reset(seed=seed)

    # init agent :
    agent: AgentOffPolicy = hydra.utils.instantiate(
        cfg["train"]["agent"], action_space=env.single_action_space
    )
    agent_state = agent.initialize(dummy_obs=first_obs, rng=rng)

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

    uniform_action = get_uniform_action_sample_fct(
        env.single_action_space, env.action_space
    )
    replaybuffer.sample = jax.jit(replaybuffer.sample)
    replaybuffer.insert = jax.jit(replaybuffer.insert)

    ################ TRAIN ################

    n_step_done = 0
    obs = jnp.array(first_obs)
    while n_step_done <= total_train_step:
        with Timer(name="action_selection", logger=None):
            if n_step_done < start_training_after_x_steps:
                actions = uniform_action(next(rng))
                actor_output = {"actions": actions}
            else:
                agent_state, actor_output = agent.actor_step(
                    agent_state=agent_state,
                    obs=obs,
                    key=next(rng),
                    evaluation=False,
                )
                actions = actor_output.actions
                actor_output = actor_output._asdict()
            assert jnp.logical_and(-1 <= actions, actions <= 1).all()
        event.trigger(
            "on_action_selection",
            step=n_step_done,
            **actor_output,
        )
        with Timer("env_step", logger=None):
            env_output = env.step(actions)

        with Timer("replaybuffer_insert", logger=None):
            new_obs, reward, terminated, truncated, info = env_output
            step_data = (obs, actions, reward, new_obs, terminated)
            buffer_state = replaybuffer.insert(buffer_state, step_data)

        if (
            n_step_done > start_training_after_x_steps
            and replaybuffer.size(buffer_state) > batch_size
        ):
            with Timer("learn_step", logger=None):
                # TODO : optimize this, sample every sample in one go and then scan
                # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                for _ in range(grad_steps_per_step * num_envs):
                    buffer_state, buffer_sample = replaybuffer.sample(buffer_state)
                    agent_state, learner_output = agent.learner_step(
                        agent_state, buffer_sample, next(rng)
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
        obs = new_obs
        n_step_done += env.num_envs

    total = sum(Timer.timers.total(name) for name in Timer.timers.data)
    print("total timers : ", total)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # to remove once package created :
    sys.path.append(os.getcwd())

    train()
