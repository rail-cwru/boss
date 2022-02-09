from environment.HierarchicalEnvironment import HierarchicalEnvironment
from config.config import ConfigItemDesc
from typing import Dict, List, Tuple, TYPE_CHECKING, Union
import numpy as np
from domain.features import BinaryFeature
from domain.observation import ObservationDomain
from domain.actions import DiscreteAction
from domain import DiscreteActionDomain
from domain.ActionDomain import ActionDomain
from domain.hierarchical_domain import HierarchicalActionDomain, action_hierarchy_from_config
import random
from common.properties import Properties
from common.domain_transfer import DomainTransferMessage


def _valid_action_hierarchy(ah):
    # TODO
    return True


class IkeaChair(HierarchicalEnvironment):

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc(name="action_hierarchy",
                           check=_valid_action_hierarchy,
                           info='The hierarchy of actions that can be executed by the agent',
                           nestable=True)
        ]

    def __init__(self, config: 'Config'):
        self.offline = True if hasattr(config, "sampler") else False
        config = config.find_config_for_instance(self)

        self.agent_class = 0
        self._flat = True

        self.n = 1 # Num agents
        self._agent_id_list = list(range(self.n))
        self._agent_class_map = {agent: self.agent_class for agent in self._agent_id_list}

        self.legs = 2
        self.stabilizers = 1
        self.cushions = 1
        self.backs = 1

        self.aligned_backs = [0 for i in range(self.backs)]
        self.connected_backs = [0 for i in range(self.backs)]

        self.aligned_cushions = [0 for i in range(self.cushions)]
        self.connected_cushions = [0 for i in range(self.cushions)]

        self.aligned_stabilizers = [0 for i in range(self.stabilizers)]
        self.connected_stabilizers = [0 for i in range(self.stabilizers)]

        self.aligned_legs = [0 for i in range(self.legs)]
        self.connected_legs = [0 for i in range(self.legs)]

        self.num_actions = 2 * (self.legs + self.stabilizers + self.cushions + self.backs)
        self.num_state_var = self.num_actions

        self.state = np.zeros(self.num_actions)
        self.default_reward = -1

        self.aligned_legs_start = 0
        self.connected_legs_start = self.legs

        self.aligned_stabilizers_start = self.connected_legs_start + self.legs
        self.connected_stabilizers_start = self.aligned_stabilizers_start + self.stabilizers

        self.aligned_cushions_start = self.connected_stabilizers_start + self.stabilizers
        self.connected_cushions_start = self.aligned_cushions_start + self.cushions

        self.aligned_back_start = self.connected_cushions_start + self.cushions
        self.connected_backs_start = self.aligned_back_start + self.backs

        config.seed = 2
        np.random.seed()
        super(IkeaChair, self).__init__(config)

        hierarchy_config_dict = config.action_hierarchy
        if hierarchy_config_dict:
            action_hierarchy = action_hierarchy_from_config(hierarchy_config_dict)
            self._flat = False
        # If applicable, calculate the number of values each state variable in the hierarchy can take on
        # This is used for grounding parameterized actions in the HierarchicalActionDomain
        if not self._flat:
            self.hierarchy = self.load_hierarchy(action_hierarchy)
            self.hierarchical_action_domain = HierarchicalActionDomain(self.hierarchy.root, self.hierarchy)
            self.hierarchical_observation_domain = self.abstract_all_observation_domains()

    def _set_environment_init_state(self) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """
        self.aligned_backs = [0 for i in range(self.backs)]
        self.connected_backs = [0 for i in range(self.backs)]

        self.aligned_cushions = [0 for i in range(self.cushions)]
        self.connected_cushions = [0 for i in range(self.cushions)]

        self.aligned_stabilizers = [0 for i in range(self.stabilizers)]
        self.connected_stabilizers = [0 for i in range(self.stabilizers)]

        self.aligned_legs = [0 for i in range(self.legs)]
        self.connected_legs = [0 for i in range(self.legs)]

        self.state = np.zeros(self.num_actions)

        return self.state

    def _create_observation_domains(self, config) -> Dict[int, ObservationDomain]:
        """
        :param config:
        :return:
        """

        items = []
        for leg in range(self.legs):
            a_leg = BinaryFeature(name=f'{leg}',prefix='aligned_leg')
            items.extend([a_leg])

            c_leg = BinaryFeature(name=f'{leg}', prefix='connected_leg')
            items.extend([c_leg])

        for stabilizer in range(self.stabilizers):
            a_stab = BinaryFeature(name=f'{stabilizer}',prefix='aligned_stabilizer')
            items.extend([a_stab])

            c_stab = BinaryFeature(name=f'{stabilizer}', prefix='connected_stabilizer')
            items.extend([c_stab])

        for cushion in range(self.cushions):
            a_cush = BinaryFeature(name=f'{cushion}',prefix='aligned_cushion')
            items.extend([a_cush])

            c_cush = BinaryFeature(name=f'{cushion}', prefix='connected_cushion')
            items.extend([c_cush])

        for back in range(self.backs):
            a_back = BinaryFeature(name=f'{back}',prefix='aligned_back')
            items.extend([a_back])

            c_back = BinaryFeature(name=f'{back}', prefix='connected_back')
            items.extend([c_back])

        self._observation_domain = ObservationDomain(items, num_agents=self.n)
        return {self.agent_class: self._observation_domain}

    def abstracted_observation_domain(self, state_variables: set) -> ObservationDomain:
        """
        Observation domain contains the state variables for Taxi World
        Every taxi can see every other taxi, passenger, and passenger status
        Holds only the state variables that are required at this node in the hierarchy
        :return: Observation Domain
        """

        items = []

        if 'connected_leg' in state_variables:
            for leg in range(self.legs):
                a_leg = BinaryFeature(name=f'{leg}', prefix='aligned_leg')
                items.extend([a_leg])

                c_leg = BinaryFeature(name=f'{leg}', prefix='connected_leg')
                items.extend([c_leg])

        if 'connected_stabilizer' in state_variables:
            for stabilizer in range(self.stabilizers):
                a_stab = BinaryFeature(name=f'{stabilizer}', prefix='aligned_stabilizer')
                items.extend([a_stab])

                c_stab = BinaryFeature(name=f'{stabilizer}', prefix='connected_stabilizer')
                items.extend([c_stab])

        if 'connected_cushion' in state_variables:
            for cushion in range(self.cushions):
                a_cush = BinaryFeature(name=f'{cushion}', prefix='aligned_cushion')
                items.extend([a_cush])

                c_cush = BinaryFeature(name=f'{cushion}', prefix='connected_cushion')
                items.extend([c_cush])

        if 'connected_back' in state_variables:
            for back in range(self.backs):
                a_back = BinaryFeature(name=f'{back}', prefix='aligned_back')
                items.extend([a_back])

                c_back = BinaryFeature(name=f'{back}', prefix='connected_back')
                items.extend([c_back])

        return ObservationDomain(items, num_agents=self.n)

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """

        # self.legs = 2
        # self.stabilizers = 2
        # self.cushions = 1
        # self.backs = 1
        connect_action = False
        reset = False

        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('chair_actions')
        for agent in actions:
            action = actions[agent][action_index][0]
            reward = self.default_reward
            action = min(action, 11)
            if action >= self.connected_backs_start:
               index = action - self.connected_backs_start
               if self.aligned_backs[index] == 1:
                   self.connected_backs[index] = 1
               else: # Tried to connect non-aligned piece
                   reset = True

               reward *= 2
               connect_action = True

            elif action >= self.aligned_back_start:
                index = action - self.aligned_back_start
                connected_cushions = True
                for i in self.connected_cushions:
                    if i == 0:
                        connected_cushions = False
                        break
                self.aligned_backs[index] = 1 if connected_cushions else 0
                reward *= 2

            elif action >= self.connected_cushions_start:
                index = action - self.connected_cushions_start
                if self.aligned_cushions[index] == 1:
                    self.connected_cushions[index] = 1
                else: # Tried to connect non-aligned piece
                    reset = True
                connect_action = True
                reward *= 4

            elif action >= self.aligned_cushions_start:
                index = action - self.aligned_cushions_start
                connected_stabilizers = True
                for i in self.connected_stabilizers:
                    if i == 0:
                        connected_stabilizers = False
                        break
                self.aligned_cushions[index] = 1 if connected_stabilizers else 0
                reward *= 4

            elif action >= self.connected_stabilizers_start:
                index = action - self.connected_stabilizers_start

                # need all legs connected and stabilizer aligned
                if self.aligned_stabilizers[index] == 1:
                    self.connected_stabilizers[index] = 1
                else:
                    reset = True
                connect_action = True
                reward *= 8

            elif action >= self.aligned_stabilizers_start:
                index = action - self.aligned_stabilizers_start
                connected_legs = True
                for i in self.connected_legs:
                    if i == 0:
                        connected_legs = False
                        break
                if connected_legs:
                    self.aligned_stabilizers[index] = 1
                reward *= 8

            elif action >= self.connected_legs_start:
                index = action - self.connected_legs_start
                if self.aligned_legs[index] == 1:
                    self.connected_legs[index] = 1
                else:
                    reset = True

                connect_action = True
                reward *= 16

            else:
                index = action - self.aligned_legs_start
                self.aligned_legs[index] = 1
                reward *= 16

            if connect_action:
                self.check_all_aligned()
            if reset:
                # print('Reset', action)
                # self._reset_state()
                self.soft_reset()
            else:
                d = True
                for i in self.connected_backs:
                    if i == 0:
                        d = False
                self.done = d

            if self.done:
                rewards[agent] = 0
            else:
                rewards[agent] = reward

        return rewards

    def soft_reset(self):
        """
        Initialize the state with the config.
        :return: Initial state
        """

        self.aligned_backs = [0 for i in range(self.backs)]
        self.connected_backs = [0 for i in range(self.backs)]

        self.aligned_cushions = [0 for i in range(self.cushions)]
        self.connected_cushions = [0 for i in range(self.cushions)]

        self.aligned_stabilizers = [0 for i in range(self.stabilizers)]
        self.connected_stabilizers = [0 for i in range(self.stabilizers)]
        #
        # self.aligned_legs = [0 for i in range(self.legs)]
        # self.connected_legs = [0 for i in range(self.legs)]
        #
        self.state = self.make_state()

    def make_state(self):
        state = np.zeros(self.num_actions)
        state[self.aligned_legs_start:self.connected_legs_start] = self.aligned_legs
        state[self.connected_legs_start: self.aligned_stabilizers_start] = self.connected_legs

        state[self.aligned_stabilizers_start:self.connected_stabilizers_start] = self.aligned_stabilizers
        state[self.connected_stabilizers_start: self.aligned_cushions_start] = self.connected_stabilizers

        state[self.aligned_cushions_start:self.connected_cushions_start] = self.aligned_cushions
        state[self.connected_cushions_start: self.aligned_back_start] = self.connected_cushions

        state[self.aligned_back_start:self.connected_backs_start] = self.aligned_backs
        state[self.connected_backs_start:] = self.connected_backs
        return state

    def check_all_aligned(self):
        for i in range(self.legs):
            self.aligned_legs[i] = self.connected_legs[i]

        for i in range(self.backs):
            self.aligned_backs[i] = self.connected_backs[i]

        for i in range(self.cushions):
            self.aligned_cushions[i] = self.connected_cushions[i]

        for i in range(self.stabilizers):
            self.aligned_stabilizers[i] = self.connected_stabilizers[i]

    def observe(self, obs_groups=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        """
        :return:
        """

        observations = {}

        # Other than the taxi location, every agent gets the same observation
        pre_observation = self._observation_domain.generate_empty_array()

        for leg in range(self.legs):
            a_leg_slice = self._observation_domain.index_for_name(name=f'{leg}', prefix='aligned_leg')
            pre_observation[a_leg_slice] = self.aligned_legs[leg]

            c_leg_slice = self._observation_domain.index_for_name(name=f'{leg}', prefix='connected_leg')
            pre_observation[c_leg_slice] = self.connected_legs[leg]

        for stabilizer in range(self.stabilizers):
            a_stabilizer_slice = self._observation_domain.index_for_name(name=f'{stabilizer}', prefix='aligned_stabilizer')
            pre_observation[a_stabilizer_slice] = self.aligned_stabilizers[stabilizer]

            c_stabilizer_slice = self._observation_domain.index_for_name(name=f'{stabilizer}', prefix='connected_stabilizer')
            pre_observation[c_stabilizer_slice] = self.connected_stabilizers[stabilizer]

        for cushion in range(self.cushions):
            a_cushion_slice = self._observation_domain.index_for_name(name=f'{cushion}', prefix='aligned_cushion')
            pre_observation[a_cushion_slice] = self.aligned_cushions[cushion]

            c_cushion_slice = self._observation_domain.index_for_name(name=f'{cushion}', prefix='connected_cushion')
            pre_observation[c_cushion_slice] = self.connected_cushions[cushion]

        for back in range(self.backs):
            a_back_slice = self._observation_domain.index_for_name(name=f'{back}', prefix='aligned_back')
            pre_observation[a_back_slice] = self.aligned_backs[back]

            c_back_slice = self._observation_domain.index_for_name(name=f'{back}', prefix='connected_back')
            pre_observation[c_back_slice] = self.connected_backs[back]

        for agent in self.agent_id_list:
            observation = pre_observation.copy()
            observations[agent] = observation

        return observations

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """

        self.aligned_backs = [0 for i in range(self.backs)]
        self.connected_backs = [0 for i in range(self.backs)]

        self.aligned_cushions = [0 for i in range(self.cushions)]
        self.connected_cushions = [0 for i in range(self.cushions)]

        self.aligned_stabilizers = [0 for i in range(self.stabilizers)]
        self.connected_stabilizers = [0 for i in range(self.stabilizers)]

        self.aligned_legs = [0 for i in range(self.legs)]
        self.connected_legs = [0 for i in range(self.legs)]

        self.state = np.zeros(self.num_actions)

        return self.state

    def load_hierarchy(self, action_hierarchy):

        self._flat = False

        state_var_values = {}
        actions = action_hierarchy.actions

        # Anything that can be passed as a parameter contributes to the number of grounded actions
        # To save memory, only enumerate values for state variables mentioned in params
        possible_params = set()

        for edge_variables in action_hierarchy.edges.values():
            for variable in edge_variables:
                possible_params.add(variable)

        bound_variables_map = {}
        self._set_environment_init_state()

        # Each param multiplies the number of ground actions that each parameterized action needs to be split into
        # by a factor of the number of values the parameter can take on
        # Create a dictionary mapping params of each action to their possible values
        for action in actions:
            grounded_actions = {}
            if not actions[action]['primitive']:
                for param in actions[action]['params']:
                    grounded_actions[param] = {}
                    for sv in actions[action]['params'][param]:
                        grounded_actions[param][sv] = state_var_values[sv]
                actions[action]['grounded_actions'] = grounded_actions

        return action_hierarchy.compile(state_var_values, bound_variables_map)

    def _create_action_domains(self, config) -> Dict[int, ActionDomain]:
        """
        Action domain contains a discrete action item of whether to move North East South or West according to the index
        it also contains two actions to interact with the passenger: 'pickup' and 'putdown'
        :param config:
        :return:
        """
        chair_actions = DiscreteAction(name='chair_actions', num_actions=self.num_actions)
        return {self.agent_class: DiscreteActionDomain([chair_actions], self.n)}

    def set_state(self, state: list):
        raise NotImplementedError

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[int]:
        return [0]

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    def set_seed(self, seed: int):
        # TODO: Does this have an additional RV?
        pass

    def set_initial_seed(self, seed: int):
        # This env does not use randomness at all
        pass

    def get_seed_state(self):
        return []

    def set_seed_state(self, seed_state):
        pass

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def transfer_domain(self, message: DomainTransferMessage) -> DomainTransferMessage:
        pass

    def visualize(self):
        raise NotImplementedError
