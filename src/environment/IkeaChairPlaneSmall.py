
from config.config import ConfigItemDesc
from typing import Dict, List, Tuple, TYPE_CHECKING, Union
import numpy as np
from domain.features import BinaryFeature, DiscreteFeature
from domain.observation import ObservationDomain
from domain.actions import DiscreteAction
from domain import DiscreteActionDomain
from domain.ActionDomain import ActionDomain
from domain.hierarchical_domain import HierarchicalActionDomain, action_hierarchy_from_config
import random
from common.properties import Properties
from common.domain_transfer import DomainTransferMessage
from environment.IkeaChair import IkeaChair
from environment.HierarchicalEnvironment import HierarchicalEnvironment
from enum import Enum
import copy


def _valid_action_hierarchy(ah):
    # TODO
    return True


class IkeaChairPlaneSmall(HierarchicalEnvironment):

    class Location(Enum):
        LEGS = 0
        STABILIZERS = 1
        CUSHIONS = 2
        BACKS = 3
        CONNECTED = 4
        # EMPTY = 5

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc(name="action_hierarchy",
                           check=_valid_action_hierarchy,
                           info='The hierarchy of actions that can be executed by the agent',
                           nestable=True),
            ConfigItemDesc(name="single_putdown",
                           check=bool,
                           info='If False, parameterize putdown action',
                           optional=True,
                           default=False),
            ConfigItemDesc(name="single_pickup",
                           check=bool,
                           info='If False, parameterize putdown action',
                           optional=True,
                           default=False),

        ]

    def __init__(self, config: 'Config'):
        self.offline = True if hasattr(config, "sampler") else False
        config = config.find_config_for_instance(self)

        self.agent_class = 0
        self._flat = True

        self.n = 1 # Num agents
        self._agent_id_list = list(range(self.n))
        self._agent_class_map = {agent: self.agent_class for agent in self._agent_id_list}

        self.x_range = 4

        self.legs = 1
        self.stabilizers = 0
        self.cushions = 1
        self.backs = 1

        self.back_x = 0
        self.leg_x = 0
        self.cushion_x = 0
        self.stabilizer_x = 0
        self.claw_x = 0
        self.claw_holding = 3 # Empty handed

        self.leg_x_vals = 2

        self.single_putdown = config.single_putdown
        self.single_pickup = config.single_pickup

        # self.connected_backs = [0 for i in range(self.backs)]
        # self.connected_cushions = [0 for i in range(self.cushions)]
        # self.connected_stabilizers = [0 for i in range(self.stabilizers)]

        # self.connected_legs = [0 for i in range(self.legs)]

        if self.single_putdown and self.single_pickup:
            print('Single Putdown and Pickup Actions!')
            self.num_actions = (self.legs + self.stabilizers + self.cushions + self.backs) + 4
        elif self.single_putdown or self.single_pickup:
            print('Single Action!')
            self.num_actions = 2 * (self.legs + self.stabilizers + self.cushions + self.backs) + 3
        else:
            print('Separate Actions')
            self.num_actions = 3 * (self.legs + self.stabilizers + self.cushions + self.backs) + 2

        self.num_state_var = self.legs + self.stabilizers + self.cushions + self.backs + 2

        self.state = np.zeros(self.num_state_var)
        self.default_reward = -1

        # self.state_vars = 1 + self.legs + self.stabilizers + self.cushions + self.backs

        config.seed = 2
        np.random.seed()
        super(IkeaChairPlaneSmall, self).__init__(config)

        hierarchy_config_dict = config.action_hierarchy
        if hierarchy_config_dict:
            action_hierarchy = action_hierarchy_from_config(hierarchy_config_dict)
            self._flat = False
        # If applicable, calculate the number of values each state variable in the hierarchy can take on
        # This is used for grounding parameterized actions in the HierarchicalActionDomain
        if not self._flat:
            self.primitive_domain = True
            self.hierarchy = self.load_hierarchy(action_hierarchy)
            self.hierarchical_action_domain = HierarchicalActionDomain(self.hierarchy.root, self.hierarchy)
            self.hierarchical_observation_domain = self.abstract_all_observation_domains()

    def _set_environment_init_state(self) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """

        self.state = np.zeros(self.num_state_var)

        x_vals = [i + 1 for i in range(self.x_range - 1)]

        random_choices = np.random.choice(x_vals, replace=False, size=self.x_range - 1)

        self.parts_x_values = random_choices

        # self.back_x = random.randrange(self.x_range)
        self.back_x = random_choices[2]
        self.state[2] = self.back_x

        self.cushion_x = random_choices[1]
        self.state[1] = self.cushion_x

        # self.stabilizer_x = random_choices[0]
        # self.state[1] = self.stabilizer_x

        # TODO: has to start at 0 or else the state space needs to be doubled (i.e need to be able to swap or add swap action)
        self.leg_x = random_choices[0]
        self.state[0] = self.leg_x

        # Random start
        self.claw_x = random.randint(0, self.x_range-1)
        self.state[3] = self.claw_x

        self.claw_holding = 3
        self.state[4] = self.claw_holding

        # print('######################################################')

        return self.state

    def abstract_all_observation_domains(self):
        h_obs_domain = {}
        for action in self.hierarchy.actions:
            if not self.hierarchy.actions[action]['primitive'] or \
                    (self.primitive_domain and 'state_variables' in self.hierarchy.actions[action]):
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.hierarchy.actions[action]['state_variables'])

        return h_obs_domain

    def abstract_all_second_observation_domains(self):
        h_obs_domain = {}
        for action in self.derived_hierarchy.actions:
            if not self.derived_hierarchy.actions[action]['primitive'] or \
                    (self.primitive_domain and 'state_variables' in self.derived_hierarchy.actions[action]):
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.derived_hierarchy.actions[action]['state_variables'])

        return h_obs_domain

    def _create_observation_domains(self, config) -> Dict[int, ObservationDomain]:
        """
        :param config:
        :return:
        """

        items = []
        # Leg can either be connected, in the claw or at 0
        a_leg = DiscreteFeature(name='leg_loc',
                                size=self.x_range + 1,
                                # size=self.leg_x_vals,
                                starts_from=0)
        items.extend([a_leg])

        # a_stab = DiscreteFeature(name='stabilizer_loc',
        #                          size=self.x_range + 1,
        #                          starts_from=0)
        # items.extend([a_stab])

        a_cush = DiscreteFeature(name='cushion_loc',
                                 size=self.x_range + 1,
                                 starts_from=0)
        items.extend([a_cush])

        a_back = DiscreteFeature(name='back_loc',
                                 size=self.x_range + 1,
                                 starts_from=0)
        items.extend([a_back])

        claw_loc = DiscreteFeature(name="claw_loc",
                                   size=self.x_range,
                                   starts_from=0)
        items.extend([claw_loc])

        claw_hold = DiscreteFeature(name="claw_holding",
                                    size=4,
                                    starts_from=0)
        items.extend([claw_hold])

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

        if 'leg_loc' in state_variables:
            a_leg = DiscreteFeature(name='leg_loc',
                                    size=self.x_range + 1,
                                    # size=self.leg_x_vals,
                                    starts_from=0)
            items.extend([a_leg])

        # if 'stabilizer_loc' in state_variables:
        #     a_stab = DiscreteFeature(name='stabilizer_loc',
        #                              size=self.x_range + 1,
        #                              starts_from=0)
        #     items.extend([a_stab])

        if 'cushion_loc' in state_variables:
            a_cush = DiscreteFeature(name='cushion_loc',
                                     size=self.x_range + 1,
                                     starts_from=0)
            items.extend([a_cush])

        if 'back_loc' in state_variables:
            a_back = DiscreteFeature(name='back_loc',
                                     size=self.x_range + 1,
                                     starts_from=0)
            items.extend([a_back])

        if "claw_loc" in state_variables:
            claw_loc = DiscreteFeature(name="claw_loc",
                                       size=self.x_range,
                                       starts_from=0)
            items.extend([claw_loc])

        if "claw_holding" in state_variables:
            claw_hold = DiscreteFeature(name="claw_holding",
                                        size=4,
                                        starts_from=0)
            items.extend([claw_hold])

        return ObservationDomain(items, num_agents=self.n)

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:

        """
        :param actions:
        :return:
        """

        connect_action = False
        reset = False

        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('chair_actions')
        for agent in actions:
            action = actions[agent][action_index][0]
            reward = self.default_reward

            # Connect
            if action < 3:

                # i) Claw must be holding the proper piece
                claw_holding = self.claw_holding == action

                # ii) All previous pieces must be attached already or action is 0

                previous_connected = all([ i ==  IkeaChairPlaneSmall.Location.CONNECTED.value for i in self.parts_x_values[0:action]])

                # IkeaChairPlane.Location.CONNECTED.value

                # iii) Claw must be at 0 (self.claw_x == 0)
                # iv) Part must already not be connected
                # already_connected = self.parts_x_values[action] != -1

                already_connected = self.parts_x_values[action] == IkeaChairPlaneSmall.Location.CONNECTED.value

                # print(already_connected, previous_connected, self.claw_holding, action, self.claw_x)

                # claw_holding = all([ val != IkeaChairPlane.Location.CLAW.value for val in self.parts_x_values])
                if claw_holding and self.claw_x == 0 and not already_connected and (action == 0 or previous_connected):
                    self.parts_x_values[action] = IkeaChairPlaneSmall.Location.CONNECTED.value
                    # print("Connected", action)
                    reward = 1
                    self.claw_holding = 3
            # elif self.single_pickup and action == 3:
            #
            #     claw_at_legs = self.claw_x == self.parts_x_values[0] and self.parts_x_values[0] < self.leg_x_vals - 1
            #     claw_at_one = self.claw_x == self.parts_x_values[1]
            #     claw_at_two = self.claw_x == self.parts_x_values[2]
            #     claw_at_three = self.claw_x == self.parts_x_values[3]
            #
            #     claw_loc = [claw_at_legs, claw_at_one, claw_at_two, claw_at_three]
            #
            #     if self.claw_holding == 4 and any(claw_loc):
            #         holding = claw_loc.index(True)
            #         self.claw_holding = holding
            #         # print('pickup', holding)
            #
            #     else:
            #         reward = -5


            # Pickup Legs
            elif action == 3:
                # Claw at legs
                claw_at_legs = self.claw_x == self.parts_x_values[0]
                # does not update location, instead will keep "old" position to return it
                if self.claw_holding == 3 and claw_at_legs:
                    # Claw is not carrying anything and there is a part at the current claw location
                    self.claw_holding = 0
                    # print('Pickup', self.claw_holding)

                else:
                    reward = -5

            # # Pickup stabilizer
            # elif action == 5:
            #     # does not update location, instead will keep "old" position to return it
            #     if self.claw_holding == 4 and self.claw_x == self.parts_x_values[1]:
            #         self.claw_holding = 1
            #         # print('Pickup', self.claw_holding)
            #     else:
            #         reward = -5

            # Pickup cushion
            elif action == 4:
                # does not update location, instead will keep "old" position to return it
                if self.claw_holding == 3 and self.claw_x == self.parts_x_values[1]:
                    self.claw_holding = 1
                    # print('Pickup', self.claw_holding)
                else:
                    reward = -5

            # Pickup back
            elif action == 5:
                # does not update location, instead will keep "old" position to return it
                if self.claw_holding == 3 and self.claw_x == self.parts_x_values[2]:
                    self.claw_holding = 2
                    # print('Pickup', self.claw_holding)
                else:
                    reward = -5

            # elif self.single_putdown and ((not self.single_pickup and action == 8) or (self.single_pickup and action == 5)):
            #     if self.claw_holding < 4 and self.claw_x == self.parts_x_values[self.claw_holding]:
            #         # print('Put down', self.claw_holding)
            #         self.claw_holding = 4
            #     else:
            #         reward = -5



            # Putdown legs
            # (will only be allowed to place back at the objects original location!)
            elif action == 6:
                if self.claw_holding == 0 and self.claw_x == self.parts_x_values[0]:
                    # print('Put down', self.claw_holding)
                    self.claw_holding = 3
                else:
                    reward = -5


            # # Putdown stabs
            # # (will only be allowed to place back at the objects original location!)
            # elif action == 7:
            #     if self.claw_holding == 1 and self.claw_x == self.parts_x_values[self.claw_holding]:
            #         # print('Put down', self.claw_holding)
            #         self.claw_holding = 4
            #     else:
            #         reward = -5

            # Putdown cush
            # (will only be allowed to place back at the objects original location!)
            elif action == 7:
                if self.claw_holding == 1 and self.claw_x == self.parts_x_values[1]:
                    # print('Put down', self.claw_holding)
                    self.claw_holding = 3
                else:
                    reward = -5

            elif action == 8:
                if self.claw_holding == 2 and self.claw_x == self.parts_x_values[2]:
                    # print('Put down', self.claw_holding)
                    self.claw_holding = 3
                else:
                    reward = -5

            # Move picker
            else:
                move_list = [-1, 1]
                target = self.claw_x + move_list[action - (self.num_actions - 2)]
                target = min(max(target, 0), self.x_range - 1)

                self.claw_x = target
                # reward = -1
                # print('Move claw', target)

        # if connect_action:
        #     self.check_all_aligned()
        # if reset:
        #     # print('Reset', action)
        #     # self._reset_state()
        #     self.soft_reset()

            d = True
            for i in self.parts_x_values[0:]:
                if i != IkeaChairPlaneSmall.Location.CONNECTED.value:
                    d = False
            self.done = d
            if d:
                # print(self.make_state())
                reward = 10
                # print('########################################################################')
            rewards[agent] = reward

        return rewards

    def soft_reset(self):
        """
        Initialize the state with the config.
        :return: Initial state
        """

        # self.aligned_backs = [0 for i in range(self.backs)]
        # self.connected_backs = [0 for i in range(self.backs)]
        #
        # self.aligned_cushions = [0 for i in range(self.cushions)]
        # self.connected_cushions = [0 for i in range(self.cushions)]
        #
        # self.aligned_stabilizers = [0 for i in range(self.stabilizers)]
        # self.connected_stabilizers = [0 for i in range(self.stabilizers)]
        #
        # self.aligned_legs = [0 for i in range(self.legs)]
        # self.connected_legs = [0 for i in range(self.legs)]
        #
        self.state = self.make_state()

    def make_state(self):
        state = np.zeros(self.num_state_var)
        # state[2] = self.back_x
        # state[1] = self.cushion_x
        # # state[2] = self.stabilizer_x
        # state[0] = self.leg_x
        state[0:2] = self.parts_x_values
        state[3] = self.claw_x
        state[4] = self.claw_holding
        return state

    def check_all_aligned(self):
        pass

        # for i in range(self.legs):
        #     self.aligned_legs[i] = self.connected_legs[i]
        #
        # for i in range(self.backs):
        #     self.aligned_backs[i] = self.connected_backs[i]
        #
        # for i in range(self.cushions):
        #     self.aligned_cushions[i] = self.connected_cushions[i]
        #
        # for i in range(self.stabilizers):
        #     self.aligned_stabilizers[i] = self.connected_stabilizers[i]

    def observe(self, obs_groups=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        """
        :return:
        """

        observations = {}

        # Other than the taxi location, every agent gets the same observation
        pre_observation = self._observation_domain.generate_empty_array()

        a_leg_slice = self._observation_domain.index_for_name(name='leg_loc')
        pre_observation[a_leg_slice] = self.parts_x_values[0]


        # a_stabilizer_slice = self._observation_domain.index_for_name(name='stabilizer_loc')
        # pre_observation[a_stabilizer_slice] = self.parts_x_values[1]


        a_cushion_slice = self._observation_domain.index_for_name(name='cushion_loc')
        pre_observation[a_cushion_slice] = self.parts_x_values[1]


        a_back_slice = self._observation_domain.index_for_name(name='back_loc')
        pre_observation[a_back_slice] = self.parts_x_values[2]

        claw_slice = self._observation_domain.index_for_name(name='claw_loc')
        pre_observation[claw_slice] = self.claw_x

        claw_hold_slice = self._observation_domain.index_for_name(name='claw_holding')
        pre_observation[claw_hold_slice] = self.claw_holding

        # claw_carry_slice = self._observation_domain.index_for_name(name='claw_carry')
        # pre_observation[claw_carry_slice] = self.claw_x

        for agent in self.agent_id_list:
            observation = pre_observation.copy()
            observations[agent] = observation

        return observations

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """

        self.state = np.zeros(self.num_state_var)

        x_vals = [i + 1 for i in range(self.x_range - 1)]

        random_choices = np.random.choice(x_vals, replace=False, size=self.x_range - 1)

        self.parts_x_values = random_choices

        # self.back_x = random.randrange(self.x_range)
        self.back_x = random_choices[2]
        self.state[2] = self.back_x

        self.cushion_x = random_choices[1]
        self.state[1] = self.cushion_x

        # self.stabilizer_x = random_choices[0]
        # self.state[1] = self.stabilizer_x

        # TODO: has to start at 0 or else the state space needs to be doubled (i.e need to be able to swap or add swap action)
        self.leg_x = random_choices[0]
        self.state[0] = self.leg_x

        # Random start
        self.claw_x = random.randint(0, self.x_range - 1)
        self.state[3] = self.claw_x

        self.claw_holding = 3
        self.state[4] = self.claw_holding

        # print('######################################################')

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
