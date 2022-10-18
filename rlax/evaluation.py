import jax.numpy as jnp


def eval_rollouts(env, agent, actor_state, params, rng):
    observation, info = env.reset()
    crewards = jnp.zeros(env.num_envs)
    rollout_finished = jnp.zeros(env.num_envs, dtype=bool)
    all_done = False
    while not all_done:
        actor_output, actor_state = agent.actor_step(
            params=params,
            obs=observation,
            actor_state=actor_state,
            key=next(rng),
            evaluation=True,
        )
        observation, reward, terminated, truncated, _ = env.step(actor_output.actions)
        done = jnp.logical_or(terminated, truncated)
        crewards += jnp.multiply(reward, ~rollout_finished)
        rollout_finished = jnp.logical_or(done, rollout_finished)
        all_done = rollout_finished.all()
    return crewards
