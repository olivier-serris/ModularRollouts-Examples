from typing import Any, Tuple
from gym import Space
from abc import ABC, abstractmethod
import jax.numpy as jnp

AgentState = Any
ActorOutput = Any
LearnerOutput = Any


class AgentOffPolicy(ABC):
    @abstractmethod
    def __init__(self, action_space: Space, kwargs) -> None:
        pass

    @abstractmethod
    def initialize(self, dummy_obs: jnp.array, key) -> AgentState:
        pass

    @abstractmethod
    def actor_step(
        self, agent_state: jnp.array, obs: jnp.array, key, evaluation=False
    ) -> Tuple[AgentState, ActorOutput]:
        pass

    @abstractmethod
    def learner_step(
        self, agent_state: AgentState, buffer_sample, key
    ) -> Tuple[AgentState, LearnerOutput]:
        pass

    @abstractmethod
    def learner_n_step(
        self, agent_state: AgentState, buffer_samples, key
    ) -> Tuple[AgentState, LearnerOutput]:
        pass
