
import random
from typing import Dict, List, Tuple, TYPE_CHECKING, Union
from common.domain_transfer import DomainTransferMessage
from domain import DiscreteActionDomain
from domain.actions import DiscreteAction
from domain.features import BinaryFeature
from domain.hierarchical_domain import HierarchicalActionDomain, action_hierarchy_from_config
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from common.properties import Properties
from config.config import ConfigItemDesc
from environment.HierarchicalEnvironment import HierarchicalEnvironment
import numpy as np

if TYPE_CHECKING:
    from config import Config, checks

def _valid_action_hierarchy(ah):
    # TODO
    return True

def boolean_check(bool):
    return True if bool == 'true' or bool == 'false' else False


class BitFlip(HierarchicalEnvironment):
    """
    BitFlip Domain: Flip a sequence of bits from left to right until the sequence is all zeros
    The penalty for not flipping left to right is resetting all bits to the left
    reward is -2^(num_bits - flipped bit) so that the first bit has the largest negative reward

    """

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc(name="num_bits",
                           check=lambda l: isinstance(l, list),
                           info='Number of Bits'),
            ConfigItemDesc(name="action_hierarchy",
                           check=_valid_action_hierarchy,
                           info='The hierarchy of actions that can be executed by the agent',
                           nestable=True),
            ConfigItemDesc(name="penalty",
                          check=boolean_check,
                          info='Should a badflip reset the state?',
                          optional=True,
                          default=False),
            ConfigItemDesc(name="small_penalty",
                           check=boolean_check,
                           info='Should a badflip reset the state?',
                           optional=True,
                           default=False),
            ConfigItemDesc(name="exp_penalty",
                           check=boolean_check,
                           info='Is penalty exponential?',
                           optional=True,
                           default=False),
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def __init__(self, config: 'Config'):

        self.offline = True if hasattr(config, "sampler") else False
        config = config.find_config_for_instance(self)

        self.use_penalty = config.penalty

        self.small_penalty = config.small_penalty

        self.use_penalty = False
        self.small_penalty = False

        self.exp_penalty = config.exp_penalty if hasattr(config, 'exp_penalty') else False

        self.n = 1
        self.num_bits = int(config.num_bits)
        self.agent_class = 0

        hierarchy_config_dict = config.action_hierarchy
        if hierarchy_config_dict:
            self._flat = False
        else:
            self._flat = True

        self._agent_id_list = list(range(self.n))
        self._agent_class_map = {agent: self.agent_class for agent in self._agent_id_list}

        config.seed = 2
        np.random.seed()
        super(BitFlip, self).__init__(config)
        self.reward_range = abs(-1 - (-1 * abs((2 ** (self.num_bits - 1)))))

        if hierarchy_config_dict:
            action_hierarchy = action_hierarchy_from_config(hierarchy_config_dict)
            self.hierarchy = self.load_hierarchy(action_hierarchy)

            # TODO: Verify name is root
            self.hierarchical_action_domain = HierarchicalActionDomain(self.hierarchy.root, self.hierarchy)
            self.hierarchical_observation_domain = self.abstract_all_observation_domains()

        # Define second hierarchy for flattened sampling methods
        self.derived_hierarchy = None
        self.derived_hierarchical_action_domain = None
        self.derived_hierarchical_observation_domain = None

        self.eval = False
        self.resets = 0

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

        if 'bit' in possible_params:
            bits = range(self.num_bits)
            state_var_values['bit'] = [f'_{b}' for b in bits]
            for b in bits:
                bound_variables_map[f'_{b}'] = b

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

    def _set_environment_init_state(self) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """
        self.state_val = np.zeros(self.num_bits)
        while list(self.state_val) == list(np.zeros(self.num_bits)):
            self.state_val = np.asarray([random.randint(0, 1) for i in range(0, self.num_bits)])
        return self.state_val

    def set_state(self, state: list):
        self.state_val = state
        self.eval = True

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """
        # random.seed()
        if not self.eval:
            self.state_val = np.zeros(self.num_bits)
            while list(self.state_val) == list(np.zeros(self.num_bits)):
                self.state_val = np.asarray([random.randint(0, 1) for i in range(0, self.num_bits)])
        return self.state_val

    def observe(self, obs_groups=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        """
        :return:
        """

        pre_obs = self._observation_domain.generate_empty_array()

        if self.offline:
            for b in range(self.num_bits):
                i_zero_slice = self._observation_domain.index_for_name(name=f'{b}', prefix='i_zero')
                pre_obs[i_zero_slice] = self.state_val[b]
        else:

            for b in range(self.num_bits):
                bit_slice = self._observation_domain.index_for_name(name=f'{b}', prefix='bit')
                pre_obs[bit_slice] = self.state_val[b]

            if not self._flat:
                for b in range(self.num_bits):
                    i_zero_slice = self._observation_domain.index_for_name(name=f'{b}', prefix='i_zero')
                    pre_obs[i_zero_slice] = self.state_val[b]

                zeros_to_left = 0
                for b in range(self.num_bits):
                    if b>0 and self.state_val[b-1] == 1 and zeros_to_left == 0:
                        zeros_to_left = 1
                    ztl_slice = self._observation_domain.index_for_name(name=f'{b}', prefix='all_bits_to_left')
                    pre_obs[ztl_slice] = zeros_to_left

        observations = {}
        for agent in self.agent_id_list:
            observation = pre_obs
            observations[agent] = observation

        return observations

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """
        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('flip_actions')
        for agent in actions:
            action = actions[agent][action_index][0]

            # Action is bit to flip, so flip_0 flips 0th bit
            # bits are indexed right to left --> [6 5 4 3 2 1]
            i = self.num_bits - action

            bad_flip = False
            if self.exp_penalty:
                reward = -1 * (2 ** i)
            else:
                reward = -1 * 2 * i
            # reward = -1 * (2 ** i)

            # Check state (to left of i)
            for j in range(action):
                if self.state_val[j] == 1:
                    # print('Bad Flip', self.state, action)
                    bad_flip = True
                    break

            # self.use_penalty = False
            # Set all bits j >= i to 1
            # if bad_flip and self.use_penalty:
            # bad_flip = False
            if bad_flip:
                index_1 = 0 # Full penalty
                self.resets += 1
                # print("resets: ", self.resets)

                self.state_val[index_1:action + 1] = [1 for i in range(index_1, action + 1)]
            # Flip bit
            else:
                self.state_val[action] = (self.state_val[action] + 1) % 2
            rewards[agent] = reward

        if np.count_nonzero(self.state_val) == 0:
            self.done = True
            self.eval = False

        return rewards

    def _create_observation_domains(self, config) -> Dict[int, ObservationDomain]:
        """
        :param config:
        :return:
        """

        items = []
        if self.offline:
            for p in range(self.num_bits):
                i_zero = BinaryFeature(name=f'{p}', prefix='i_zero')
                items.extend([i_zero])
        else:
            for p in range(self.num_bits):
                bit = BinaryFeature(name=f'{p}', prefix='bit')
                items.extend([bit])

            if not self._flat:
                for p in range(self.num_bits):
                    i_zero = BinaryFeature(name=f'{p}', prefix='i_zero')
                    items.extend([i_zero])

                for p in range(self.num_bits):
                    all_bits_to_left = BinaryFeature(name=f'{p}', prefix='all_bits_to_left')
                    items.extend([all_bits_to_left])

        self._observation_domain = ObservationDomain(items, num_agents=self.n)
        return {self.agent_class: self._observation_domain}

    def _create_action_domains(self, config) -> Dict[int, ActionDomain]:
        """
        Action domain contains a discrete action item of whether to move North East South or West according to the index
        it also contains two actions to interact with the passenger: 'pickup' and 'putdown'
        :param config:
        :return:
        """
        flip_actions = DiscreteAction(name='flip_actions', num_actions=self.num_bits)
        return {self.agent_class: DiscreteActionDomain([flip_actions], self.num_bits)}

    def transfer_domain(self, message: DomainTransferMessage) -> DomainTransferMessage:
        pass

    def visualize(self):
        """
        Visualize the current map
        :return:
        """
        raise NotImplementedError('Not visualizable yet... should be easy though')

    def abstract_all_observation_domains(self):
        h_obs_domain = {}
        for action in self.hierarchy.actions:
            if not self.hierarchy.actions[action]['primitive']:
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.hierarchy.actions[action]['state_variables'], action)

        return h_obs_domain

    def abstracted_observation_domain(self, state_variables: set, action=None) -> ObservationDomain:
        """
        Observation domain contains the state variables for Bitflip domain
        Holds only the state variables that are required at this node in the hierarchy
        :return: Observation Domain
        """
        items = []
        action_num = str(self.num_bits-1 if action == 'Root' else action[-1])

        if 'i_zero' in state_variables:
            # The agent's position
            i_zero = BinaryFeature(prefix='i_zero', name=action_num)
            items.extend([i_zero])

        if 'all_bits_to_left' in state_variables:
            # The list of passenger sources
            all_bits_to_left = BinaryFeature(prefix='all_bits_to_left', name=action_num)
            items.extend([all_bits_to_left])

        for var in state_variables:

            if var[-2].isdigit():
                var_num = str(var[-2]) + str(var[-1])
            else:
                var_num = str(var[-1])

            if 'i_zero_' in var:
                i_zero = BinaryFeature(prefix='i_zero', name = var_num)
                items.extend([i_zero])

            elif 'all_bits_to_left_' in var:
                all_bits_to_left = BinaryFeature(prefix='all_bits_to_left', name = var_num)
                items.extend([all_bits_to_left])

            elif 'bit_' in var:
                bit = BinaryFeature(prefix='bit', name = var_num)
                items.extend([bit])

        return ObservationDomain(items, num_agents=self.n)

    def abstract_all_second_observation_domains(self):
        h_obs_domain = {}
        for action in self.derived_hierarchy.actions:
            if not self.derived_hierarchy.actions[action]['primitive']:
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.derived_hierarchy.actions[action]['state_variables'], action)

        return h_obs_domain

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
