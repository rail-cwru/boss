from typing import List, Tuple, Dict, TYPE_CHECKING, Union

import numpy as np

from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from domain.features import Feature
from common.properties import Properties
from config.config import ConfigItemDesc
from environment import Environment
if TYPE_CHECKING:
    from config import Config, checks


class TestEnvironment(Environment):
    """
    state will maintain the number of times each agent has moved forward.
    If an agent moves to the left or right, the reward will be -1, and the agent is killed.
    If the agen moves forward reward is +1.
    The config can specify a number of times the agent can move forward before an episode ends.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc('num_agents', checks.positive_integer, 'Number of agents'),
            ConfigItemDesc('total_steps', checks.positive_integer, 'Total number of steps agent can move forward')
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[str]:
        return ['agent_class']

    @property
    def agent_id_list(self) -> List[int]:
        """
        list of all the numbers from 0 to n. one for each agent.
        :return: list of agentIds
        """
        return self._agent_id_list

    # From what I can tell this env does not use randomness
    def set_initial_seed(self, seed: int):
        pass

    def get_seed_state(self):
        return []

    def set_seed_state(self, seed_state):
        pass

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        move all agents, and remove those that died.
        :param actions: n tuple of actions. 1 will be forward, anything else will not be.
        :return: observations and rewards. Observation will be a n by 1 array. 1 means the agent has not died, 0 means it has.
        agents that were previously killed will observe 0.
                reward will be pretty much the same, but will get -1 reward on the turn an agent dies.
        """
        # update positions
        action_array = np.zeros((self.num_agents))
        for i in range(self.num_agents):
            action_array[i] = actions[i]

        #any agent that was already dead will have no action
        # Slot n of state is for the "steps left" data and doesn't correspond to agent.
        action_array[self.state[0:self.num_agents] == -1] = -1

        #update state
        # (Somewhat Memory-inefficient. But the simplicity absolves it of greater issues...?)
        self.state[0:self.num_agents][action_array == 1] += 1
        self.state[0:self.num_agents][action_array != 1] = -1
        if self.state[self.step_index] > 0:
            self.state[self.step_index] -= 1

        rewards: Dict[int, int] = {}
        #get rewards
        for i in range(self.num_agents):
            if action_array[i] == -1:
                rewards[i] = 0 # agent was already dead
            elif action_array[i] == 1:
                rewards[i] = 1
            else:
                rewards[i] = -1 #agent just died.

        # Check termination when all agents dead
        if (self.state[0:self.num_agents] == -1).all():
            self.done = True

        return rewards

    def observe(self, observation_request=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        observations: Dict[int, np.ndarray] = {}
        # get observation
        for i in range(self.num_agents):
            observation_domain = self.agent_class_observation_domains[self.agent_class]
            observations[i] = observation_domain.generate_empty_array()
            observations[i][observation_domain.index_for_name('agent_position')] = self.state[i]
            observations[i][observation_domain.index_for_name('steps_left')] = self.state[self.step_index]

        return observations

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        """
        state will have n spots for the position of n  agents, and the number of steps left to take.
        config.environment should have totalSteps, and an n.
        :return:
        """
        self.agent_class = 'agent_class'
        self.num_agents = self.config.num_agents
        self.step_index = self.num_agents
        self._agent_id_list = list(range(self.num_agents))
        self._agent_class_map = {agent_id: self.agent_class_list[0] for agent_id in self._agent_id_list}

        self.total_steps = self.config.total_steps
        state = np.zeros((self.num_agents+1))
        if self.total_steps is not None:
            state[self.step_index] = self.total_steps
        else:
            state[self.step_index] = self.total_steps = -1 #this means there is an infinite number of steps.

        return state

    # def _create_state_domain(self, config: ExperimentConfig) -> Domain:
    #     # agent_positions : DomainItem = DomainItem('agent_positions', [self.num_agents, 1], 'int', slice(-1,None))
    #     # Using a separate domain item for the location of each agent:
    #     items : List[DomainItem] = []
    #     for agent_id in self.agent_id_list:
    #         items.append(DomainItem(name=agent_id, shape=[1], dtype='int', drange=slice(-1,None)))
    #     items.append(DomainItem(name='steps_left', shape=[1], dtype='int', drange=slice(-1,None)))
    #     return Domain(items = items)

    def _create_observation_domains(self, config: 'Config') -> Dict[str, ObservationDomain]:
        agent_position : Feature = Feature(name='agent_position', shape=[1], dtype='int', drange=slice(-1, None))
        steps_left : Feature = Feature(name='steps_left', shape=[1], dtype='int', drange=slice(-1, None))
        domain = ObservationDomain([agent_position, steps_left], self.num_agents)
        self._observation_array = domain.generate_empty_array()
        return {self.agent_class_list[0]: domain}


    def _create_action_domains(self, config) -> Dict[str, ActionDomain]:
        agent_action : Feature = Feature(name='agent_action', shape=[1], dtype='int', drange=slice(0, 1))
        return {self.agent_class_list[0]: ActionDomain(items = [agent_action], num_agents=self.num_agents)}
