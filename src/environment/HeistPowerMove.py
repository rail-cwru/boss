
from environment.HierarchicalEnvironment import HierarchicalEnvironment
from domain.features import CoordinateFeature, DiscreteFeature
from domain.observation import ObservationDomain
from typing import Dict, List, Tuple, TYPE_CHECKING, Union
import numpy as np
from config.config import ConfigItemDesc
from common.domain_transfer import DomainTransferMessage
from domain.hierarchical_domain import HierarchicalActionDomain, action_hierarchy_from_config
from common.properties import Properties
from collections import defaultdict
from domain import DiscreteActionDomain
from domain.actions import DiscreteAction
from domain.ActionDomain import ActionDomain
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import random



if TYPE_CHECKING:
    from config import Config

def _valid_action_hierarchy(ah):
    # TODO
    return True

def _valid_taxiworld_tiles(data):
    assert isinstance(data, dict), 'The Taxi World tile data must be a dictionary.'
    for k, v in data.items():
        if int(k) == 0:
            assert v == 'EMPTY', 'Type 0 must always be EMPTY'
        elif int(k) == 1:
            assert v == 'BK', 'Type 1 must always be a Blue Key (BK)'
        elif int(k) == 2:
            assert v == 'BL', 'Type 2 must always be a Blue Lock (BL)'
        elif int(k) == 3:
            assert v == 'GK', 'Type 3 must always be a Green Key (GK)'
        elif int(k) == 4:
            assert v == 'GL', 'Type 4 must always be a Green Lock (GL)'
        elif int(k) == 5:
            assert v == 'OK', 'Type 5 must always be a Orange Key (OK)'
        elif int(k) == 6:
            assert v == 'OL', 'Type 6 must always be a Orange Lock (OL)'
        elif int(k) == 7:
            assert v == 'G', 'Type 7 must always be a Gem'
        else:
            # TODO remove references to special tiles? Pseudo-rewards will not be implemented here
            err_msg = 'Types other than "0" through "7" must be special tiles ' \
                      'specifying [reward] int and [terminate] boolean.'
            assert isinstance(v, dict) and 'reward' in v and 'terminate' in v, err_msg
    return True

def _valid_taxiworld_map(data):
    msg1 = 'The Taxi World map data must be a list of integer strings.'
    assert isinstance(data, list), msg1
    width = None
    for row in data:
        assert isinstance(row, str), msg1
        if width is None:
            width = len(row)
        else:
            assert len(row) == width, 'The taxiworld map data must be rectangular but the row ' \
                                      '[{}] did not match the previous width of [{}]'.format(row, width)
    return True


class HeistPowerMove(HierarchicalEnvironment):
    """

    This is a modification of the original Heist domain.
    It has an additional action, PowerMove, which can move the agent through locks and
    will always move the agent one step closer to the gem. Will unlock doors

    Has the option to add resets, which means that the environent will reset if the agent hits a locked door or wall,
    which can stimulate the agent getting caught
    """

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc(name="action_hierarchy",
                           check=_valid_action_hierarchy,
                           info='The hierarchy of actions that can be executed by the agent',
                           nestable=True),
            ConfigItemDesc(name="tile_types",
                           check=_valid_taxiworld_tiles,
                           info='Descriptions of Taxi World Tile Types.'),
            ConfigItemDesc(name='map',
                           check=_valid_taxiworld_map,
                           info='Taxi World map. A list of strings filled with symbols corresponding to tile type.'),
            ConfigItemDesc(name='agent_spawn',
                           check=lambda l: isinstance(l, list) and
                                           all([isinstance(duple, str) or
                                                (isinstance(duple, list) and len(duple) == 2 and
                                                 all([isinstance(coord, int) and coord >= 0 for coord in duple]))
                                                for duple in l]),
                           info='Agent spawning coordinates as duples of int coordinate locations or '
                                'random for random spawn.'),
            ConfigItemDesc(name='default_reward',
                           check=lambda i: isinstance(i, int),
                           info='Default reward for steps elapsed in Taxi World. Typically -1.'),
            ConfigItemDesc(name='walls',
                           check=lambda l: all([isinstance(i, tuple) for i in l]),
                           info='List of tuples holding the location of walls. Will have two numbers (1, 2),' +
                                 'which means that there is a wall between spots 1 and 2'),
            ConfigItemDesc(name='agent_spawn_possibilities',
                           check=lambda l: isinstance(l, list) and
                                           all([isinstance(duple, str) or
                                                (isinstance(duple, list) and len(duple) == 2 and
                                                 all([isinstance(coord, int) and coord >= 0 for coord in duple]))
                                                for duple in l]),
                           info='Agent spawning coordinates as duples of int coordinate locations or '
                                'random for random spawn.',
                           optional=True,
                           default=[[2, 4]]),
            ConfigItemDesc(name='reset_probability',
                           check=lambda i: isinstance(i, float),
                           info='Probability of resetting domain after powermove',
                           optional=True, default=0.0),
            ConfigItemDesc(name='lock_reset_probability',
                           check=lambda i: isinstance(i, float),
                           info='Probability of resetting domain after unlock/pickup',
                           optional=True, default=0.0),
            ConfigItemDesc(name='wall_reset_probability',
                           check=lambda i: isinstance(i, float),
                           info='Probability of resetting domain after running into wall',
                           optional=True, default=0.0),
        ]

    def __init__(self, config: 'Config'):

        self.num_agents = 1
        # self.map_size = 5
        self.stochastic = False
        self.agent_class = 0

        self.num_actions = 12
        self.n = 1

        self.move_actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        config = config.find_config_for_instance(self)

        self.walls = config.walls
        self.config_map = config.map
        self.config_agent_spawn = config.agent_spawn
        self.config_agent_spawn_possibilities = config.agent_spawn_possibilities
        self.default_reward = config.default_reward
        self.reset_probability = config.reset_probability
        self.lock_reset_probability = config.lock_reset_probability
        self.wall_reset_probability = config.wall_reset_probability

        self.tile_id_dict = defaultdict()
        self.make_tile_id_dict()

        self.bk_possible_loc = None
        self.bl_possible_loc = None
        self.gk_possible_loc = None
        self.gl_possible_loc = None
        self.ok_possible_loc = None
        self.ol_possible_loc = None
        self.gem_possible_loc = None

        self.bk_loc = None
        self.bl_loc = None
        self.gk_loc = None
        self.gl_loc = None
        self.ok_loc = None
        self.ol_loc = None
        self.gem_loc = None

        self.bk = -1
        self.bl = -1
        self.gk = -1
        self.gl = -1
        self.ok = -1
        self.ol = -1
        self.g = -1
        self.stochastic = True

        self._set_environment_init_state()

        self.num_state_var = 9

        np.random.seed()
        seed = np.random.get_state()[1][0]
        config.seed = seed

        self._flat = True
        super(HeistPowerMove, self).__init__(config)

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

        self.right_map = [
            [4, 1],
            [4, 2],
            [2, 1],
            [2, 2]
                    ]

        self.up_map = [
            [4, 3],
            [3, 3],
            [2, 3],
            [1, 4],
            [3, 1]
        ]

        self.down_map = [
            [3, 4],
            [0, 3],
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0]
        ]

        self.left_map = [
            [4, 4],
            [2, 4],
            [3, 2],
            [0, 4],
            [1, 3],
            [1, 2],
            [1, 1],
            [0, 1],
            [0, 2]
            ]

        self.reward_range = 15

    def get_spot_index(self, y: int, x: int):
        return self.cols * y + x

    def make_tile_id_dict(self):
        self.tile_id_dict['EMPTY'] = 0
        self.tile_id_dict['BK'] = 1
        self.tile_id_dict['BL'] = 2
        self.tile_id_dict['GK'] = 3
        self.tile_id_dict['GL'] = 4
        self.tile_id_dict['OK'] = 5
        self.tile_id_dict['OL'] = 6
        self.tile_id_dict['Gem'] = 7

    def abstract_all_observation_domains(self):
        h_obs_domain = {}
        for action in self.hierarchy.actions:
            if not self.hierarchy.actions[action]['primitive'] or \
                     'state_variables' in self.hierarchy.actions[action]:
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.hierarchy.actions[action]['state_variables'])

        return h_obs_domain

    def abstract_all_second_observation_domains(self):
        h_obs_domain = {}
        for action in self.derived_hierarchy.actions:
            if not self.derived_hierarchy.actions[action]['primitive'] or \
                    'state_variables' in self.derived_hierarchy.actions[action]:
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.derived_hierarchy.actions[action]['state_variables'])

        return h_obs_domain

    def _set_environment_init_state(self) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """
        self.terminated_count = 0

        # Convert String-representation of map to np array
        map_rows: List[str] = self.config_map
        self.rows = len(map_rows)
        self.cols = len(map_rows[0])
        self.map = np.zeros((self.rows, self.cols))
        for y, row in enumerate(map_rows):
            for x, item in enumerate(row):
                self.map[y, x] = int(item)

        self.n = len(self.config_agent_spawn)
        self._agent_id_list = list(range(self.n))
        self._agent_class_map = {agent: self.agent_class for agent in self._agent_id_list}

        # Convert agent_spawn to dictionary
        self._agent_positions: Dict[int, List[int, int]] = {}
        for agent, position in enumerate(self.config_agent_spawn):
            if position == 'random':
                raise NotImplementedError("Cannot pick random location")
            elif position == 'list':
                self._agent_positions[agent] = random.choice(self.config_agent_spawn_possibilities)
            else:
                self._agent_positions[agent] = position

        self.bk_possible_loc = np.argwhere(self.map == 1).tolist()
        self.bl_possible_loc = np.argwhere(self.map == 2).tolist()
        self.gk_possible_loc = np.argwhere(self.map == 3).tolist()
        self.gl_possible_loc = np.argwhere(self.map == 4).tolist()
        self.ok_possible_loc = np.argwhere(self.map == 5).tolist()
        self.ol_possible_loc = np.argwhere(self.map == 6).tolist()
        self.gem_possible_loc = np.argwhere(self.map == 7).tolist()

        self.bk_loc = random.choice(self.bk_possible_loc)
        self.bl_loc = random.choice(self.bl_possible_loc)
        self.gk_loc = random.choice(self.gk_possible_loc)
        self.gl_loc = random.choice(self.gl_possible_loc)
        self.ok_loc = random.choice(self.ok_possible_loc)
        self.ol_loc = random.choice(self.ol_possible_loc)
        self.gem_loc = random.choice(self.gem_possible_loc)

        self.bk = self.bk_possible_loc.index(self.bk_loc)
        self.bl = self.bl_possible_loc.index(self.bl_loc)
        self.gk = self.gk_possible_loc.index(self.gk_loc)
        self.gl = self.gl_possible_loc.index(self.gl_loc)
        self.ok = self.ok_possible_loc.index(self.ok_loc)
        self.ol = self.ol_possible_loc.index(self.ol_loc)
        self.g = self.gem_possible_loc.index(self.gem_loc)

    def observe(self, obs_groups=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        """
        :return:
        """
        observations = {}

        # Other than the taxi location, every agent gets the same observation
        pre_observation = self._observation_domain.generate_empty_array()

        p_loc = self._observation_domain.index_for_name(name='agent_loc')
        pre_observation[p_loc] = self._agent_positions[0]

        bk_loc = self._observation_domain.index_for_name(name='blue_key_loc')
        pre_observation[bk_loc] = self.bk

        bl_loc = self._observation_domain.index_for_name(name='blue_lock_loc')
        pre_observation[bl_loc] = self.bl

        gk_loc = self._observation_domain.index_for_name(name='green_key_loc')
        pre_observation[gk_loc] = self.gk

        gl_loc = self._observation_domain.index_for_name(name='green_lock_loc')
        pre_observation[gl_loc] = self.gl

        ok_loc = self._observation_domain.index_for_name(name='orange_key_loc')
        pre_observation[ok_loc] = self.ok

        ol_loc = self._observation_domain.index_for_name(name='orange_lock_loc')
        pre_observation[ol_loc] = self.ol

        gem_loc = self._observation_domain.index_for_name(name='gem_loc')
        pre_observation[gem_loc] = self.g

        for agent in self.agent_id_list:
            observation = pre_observation.copy()
            observations[agent] = observation

        return observations

    def set_seed(self, seed: int):
        # TODO: Does this have an additional RV?
        pass

    # This env does not use randomness at all
    def set_initial_seed(self, seed: int):
        pass

    def get_seed_state(self):
        return []

    def set_seed_state(self, seed_state):
        pass

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """
        self.terminated_count = 0

        # Convert String-representation of map to np array
        map_rows: List[str] = self.config_map
        self.rows = len(map_rows)
        self.cols = len(map_rows[0])
        self.map = np.zeros((self.rows, self.cols))
        for y, row in enumerate(map_rows):
            for x, item in enumerate(row):
                self.map[y, x] = int(item)

        self.n = len(self.config_agent_spawn)
        self._agent_id_list = list(range(self.n))
        self._agent_class_map = {agent: self.agent_class for agent in self._agent_id_list}

        # Convert agent_spawn to dictionary
        self._agent_positions: Dict[int, List[int, int]] = {}
        for agent, position in enumerate(self.config_agent_spawn):
            if position == 'random':
                raise NotImplementedError("Cannot pick random location")
            elif position == 'list':
                self._agent_positions[agent] = random.choice(self.config_agent_spawn_possibilities)
            else:
                self._agent_positions[agent] = position

        self.bk_possible_loc = np.argwhere(self.map == 1).tolist()
        self.bl_possible_loc = np.argwhere(self.map == 2).tolist()
        self.gk_possible_loc = np.argwhere(self.map == 3).tolist()
        self.gl_possible_loc = np.argwhere(self.map == 4).tolist()
        self.ok_possible_loc = np.argwhere(self.map == 5).tolist()
        self.ol_possible_loc = np.argwhere(self.map == 6).tolist()
        self.gem_possible_loc = np.argwhere(self.map == 7).tolist()

        self.bk_loc = random.choice(self.bk_possible_loc)
        self.bl_loc = random.choice(self.bl_possible_loc)
        self.gk_loc = random.choice(self.gk_possible_loc)
        self.gl_loc = random.choice(self.gl_possible_loc)
        self.ok_loc = random.choice(self.ok_possible_loc)
        self.ol_loc = random.choice(self.ol_possible_loc)
        self.gem_loc = random.choice(self.gem_possible_loc)

        self.bk = self.bk_possible_loc.index(self.bk_loc)
        self.bl = self.bl_possible_loc.index(self.bl_loc)
        self.gk = self.gk_possible_loc.index(self.gk_loc)
        self.gl = self.gl_possible_loc.index(self.gl_loc)
        self.ok = self.ok_possible_loc.index(self.ok_loc)
        self.ol = self.ol_possible_loc.index(self.ol_loc)
        self.g = self.gem_possible_loc.index(self.gem_loc)

    def _create_observation_domains(self, config) -> Dict[int, ObservationDomain]:
        """
        Observation domain contains the state variables for Taxi World
        Every taxi can see every other taxi, passenger, and passenger status
        holding_passenger takes several values:
            -1 is initial value, passenger has not been picked up yet
            0...n-1 is the taxi that is currently holding the passenger
            -2 ... -n - 1 is the taxi that was holding the passenger before it was dropped off
                NOTE: -2 is taxi 0
        :param config:
        :return:
        """
        items = []
        # The agent's position
        agent_loc = CoordinateFeature(name='agent_loc', lower=[0, 0],
                                      upper=[self.rows, self.cols],
                                      is_discrete=True)
        items.extend([agent_loc])

        bk_loc = DiscreteFeature(name='blue_key_loc', size=len(self.bk_possible_loc )+ 1, starts_from=0)
        items.extend([bk_loc])

        bl_loc = DiscreteFeature(name='blue_lock_loc', size=len(self.bl_possible_loc )+ 1, starts_from=0)
        items.extend([bl_loc])

        gk_loc = DiscreteFeature(name='green_key_loc', size=len(self.gk_possible_loc )+ 1, starts_from=0)
        items.extend([gk_loc])

        gl_loc = DiscreteFeature(name='green_lock_loc', size=len(self.gl_possible_loc )+ 1, starts_from=0)
        items.extend([gl_loc])

        ok_loc = DiscreteFeature(name='orange_key_loc', size=len(self.ok_possible_loc)+ 1, starts_from=0)
        items.extend([ok_loc])

        ol_loc = DiscreteFeature(name='orange_lock_loc', size=len(self.ol_possible_loc )+ 1, starts_from=0)
        items.extend([ol_loc])

        gem_loc = DiscreteFeature(name='gem_loc', size=len(self.gem_possible_loc ) + 1, starts_from=0)
        items.extend([gem_loc])

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

        if 'agent_loc' in state_variables:
            # The agent's position
            agent_loc = CoordinateFeature(name='agent_loc', lower=[0, 0],
                                          upper=[self.rows, self.cols],
                                          is_discrete=True)
            items.extend([agent_loc])

        if 'blue_key_loc' in state_variables:
            bk_loc = DiscreteFeature(name='blue_key_loc', size=len(self.bk_possible_loc )+ 1, starts_from=0)
            items.extend([bk_loc])

        if 'blue_lock_loc' in state_variables:
            bl_loc = DiscreteFeature(name='blue_lock_loc', size=len(self.bl_possible_loc ) + 1, starts_from=0)
            items.extend([bl_loc])

        if 'green_key_loc' in state_variables:
            gk_loc = DiscreteFeature(name='green_key_loc', size=len(self.gk_possible_loc ) + 1, starts_from=0)
            items.extend([gk_loc])

        if 'green_lock_loc' in state_variables:
            gl_loc = DiscreteFeature(name='green_lock_loc', size=len(self.gl_possible_loc ) + 1, starts_from=0)
            items.extend([gl_loc])

        if 'orange_key_loc' in state_variables:
            ok_loc = DiscreteFeature(name='orange_key_loc', size=len(self.ok_possible_loc )+ 1, starts_from=0)
            items.extend([ok_loc])

        if 'orange_lock_loc' in state_variables:
            ol_loc = DiscreteFeature(name='orange_lock_loc', size=len(self.ol_possible_loc ) + 1, starts_from=0)
            items.extend([ol_loc])

        if 'gem_loc' in state_variables:
            gem_loc = DiscreteFeature(name='gem_loc', size=len(self.gem_possible_loc ) + 1, starts_from=0)
            items.extend([gem_loc])

        return ObservationDomain(items, num_agents=self.n)

    def _create_action_domains(self, config) -> Dict[int, ActionDomain]:
        """
        Action domain contains a discrete action item of whether to move North East South or West according to the index
        it also contains two actions to interact with the passenger: 'pickup' and 'putdown'
        :param config:
        :return:
        """
        taxi_actions = DiscreteAction(name='heist_actions', num_actions=self.num_actions)
        return {self.agent_class: DiscreteActionDomain([taxi_actions], self.n)}

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """

        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('heist_actions')
        for agent in actions:
            action = actions[agent][action_index][0]
            reward = self.default_reward

            if action < 4:

                if self.stochastic:
                    choice1 = action - 1 if action - 1 >= 0 else 3
                    choice2 = action + 1 if action + 1 < 4 else 0
                    action = np.random.choice([choice1, action, choice2], 1, p=[0.1, 0.8, 0.1])[0]

                move = self.move_actions[action]
                current_position = self._agent_positions[agent]
                target_position = [current_position[0] + move[0], current_position[1] + move[1]]

                current_pos_ind = self.get_spot_index(current_position[0], current_position[1])
                target_pos_ind = self.get_spot_index(target_position[0], target_position[1])

                # There is a wall in the way
                if [current_pos_ind, target_pos_ind] in self.walls or [target_pos_ind, current_pos_ind] in self.walls:
                    target_position = current_position

                    if random.random() < self.wall_reset_probability:
                        self._reset_state()
                        target_position = self._agent_positions[agent]

                # Check that the target position is inside the bounds of the map
                elif target_position[0] not in range(self.rows) or target_position[1] not in range(self.cols):
                    # Outside of map, don't move
                    target_position = current_position

                    if random.random() < self.wall_reset_probability:
                        self._reset_state()
                        target_position = self._agent_positions[agent]

                # Check for move onto lock without unlocking it
                elif (self.bl == 0 and target_position == self.bl_loc) or \
                    (self.gl == 0 and target_position == self.gl_loc) or  \
                    (self.ol == 0 and target_position == self.ol_loc):

                    target_position = current_position  # Cannot move onto lock without unlocking it first

                    if random.random() < self.wall_reset_probability:
                        self._reset_state()
                        target_position = self._agent_positions[agent]

                # Move the agent
                self._agent_positions[agent] = target_position

            # Pickup Blue Key
            elif action == 4:
                if self.successfully_pickup_blue_key(agent):
                    self.bk = len(self.bk_possible_loc)
                    # print('Pick up Blue')
                else:
                    if random.random() < self.lock_reset_probability:
                        self._reset_state()
                    reward = -5

            # Pickup Green Key
            elif action == 5:
                if self.successfully_pickup_green_key(agent):
                    self.gk = len(self.gk_possible_loc)
                    # print('Pick up Green')
                else:
                    if random.random() < self.lock_reset_probability:
                        self._reset_state()
                    reward = -5

            # Pickup Orange Key
            elif action == 6:
                if self.successfully_pickup_orange_key(agent):
                    self.ok = len(self.ok_possible_loc)
                    # print('Pick up Orange')
                else:
                    if random.random() < self.lock_reset_probability:
                        self._reset_state()
                    reward = -5

            # Pickup Gem
            elif action == 7:
                if self.successfully_pickup_gem(agent):
                    # Successfully picked up gem
                    self.g = 1
                    self.done = True
                    reward = 10
                else:
                    if random.random() < self.lock_reset_probability:
                        self._reset_state()
                    reward = -5

            # Open blue lock
            elif action == 8:
                if self.successfully_unlock_blue(agent):
                    self.bl = len(self.bl_possible_loc)
                    # print('Unlock Blue')
                else:
                    if random.random() < self.lock_reset_probability:
                        self._reset_state()

                    reward = -5

            # Open green lock
            elif action == 9:
                if self.successfully_unlock_green(agent):
                    self.gl = len(self.gl_possible_loc)
                    # print('Unlock Green')
                else:
                    if random.random() < self.lock_reset_probability:
                        self._reset_state()

                    reward = -5

            # Unlock Orange
            elif action == 10:
                if self.successfully_unlock_orange(agent):
                    self.ol = len(self.ol_possible_loc)
                    # print('Unlock Orange')
                else:
                    if random.random() < self.lock_reset_probability:
                        self._reset_state()

                    reward = -5

            # PowerMove
            elif action == 11:

                current_position = self._agent_positions[agent]

                if current_position in self.left_map:
                    move = self.move_actions[2]
                elif current_position in self.up_map:
                    move = self.move_actions[0]
                elif current_position in self.down_map:
                    move = self.move_actions[1]
                elif current_position in self.right_map:
                    move = self.move_actions[3]
                else:
                    move = [0, 0]

                target_position = [current_position[0] + move[0], current_position[1] + move[1]]
                # reward = -2

                # print(current_position, target_position)

                # best_target = None
                # min_dist = 1000
                #
                # for move in self.move_actions:
                #     target_position = [current_position[0] + move[0], current_position[1] + move[1]]
                #     target_pos_ind = self.get_spot_index(target_position[0], target_position[1])
                #     # if not ([current_pos_ind, target_pos_ind] in self.walls or [target_pos_ind,
                #     #                                                        current_pos_ind] in self.walls):
                #     dist = self.manhattan_dist(gem_loc, target_position)
                #     if min_dist> dist:
                #         min_dist = dist
                #         best_target = target_position

                if target_position == self.bl_loc:
                    self.bl = 1
                    self.bk = 1
                elif target_position == self.gl_loc:
                    self.gl = 1
                    self.gk = 1
                elif target_position == self.ol_loc:
                    self.ol = 1
                    self.ok = 1
                self._agent_positions[agent] = target_position

                if random.random() < self.reset_probability:
                    self._reset_state()

            rewards[agent] = reward
        return rewards

    def manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def successfully_pickup_blue_key(self, agent) -> bool:
        return self.bk_loc == self._agent_positions[agent] and self.bk < len(self.bk_possible_loc)

    def successfully_pickup_green_key(self, agent) -> bool:
        return self.gk_loc == self._agent_positions[agent] and self.gk < len(self.gk_possible_loc)

    def successfully_pickup_orange_key(self, agent) -> bool:
        return self.ok_loc == self._agent_positions[agent] and self.ok < len(self.ok_possible_loc)

    def successfully_pickup_gem(self, agent) -> bool:
        # if self.gem_loc == self._agent_positions[agent] and self.ol != len(self.ol_possible_loc):
        #     raise ValueError("Map Not implemented right... agent got Gem before orange lock")
        return self.gem_loc == self._agent_positions[agent]

    def successfully_unlock_blue(self, agent) -> bool:
        location = self.bl_loc
        return (self.bk == len(self.bk_possible_loc) and
                self.bl < len(self.bl_possible_loc) and
                self.is_adjacent(location, self._agent_positions[agent]))

    def successfully_unlock_orange(self, agent) -> bool:
        location = self.ol_loc
        return (self.ok == len(self.ok_possible_loc) and
                self.ol < len(self.ol_possible_loc) and
                self.is_adjacent(location, self._agent_positions[agent]))

    def successfully_unlock_green(self, agent) -> bool:
        location = self.gl_loc

        return (self.gk == len(self.gk_possible_loc) and
                self.gl < len(self.gl_possible_loc) and
                self.is_adjacent(location, self._agent_positions[agent]))

    def is_adjacent(self, loc1, loc2):
        d1 = abs(loc1[0] - loc2[0])
        d2 = abs(loc1[1] - loc2[1])
        if d1 * d2 == 0 and max(d1, d2) == 1:
            return True
        else:
            return False

    def transfer_domain(self, message: DomainTransferMessage) -> DomainTransferMessage:
        pass

    def make_state(self):
        pass

    def visualize(self):
        """
        Visualize the current map
        :return:
        """
        visual = np.copy(self.map)
        for agent in self._agent_positions:
            r = self._agent_positions[agent][0]
            c = self._agent_positions[agent][1]
            visual[r, c] = 5
        cmap = ListedColormap(colors=['w', 'k', 'g', 'r', 'y', 'b'])
        if not hasattr(self, '_image'):
            fig = plt.figure('env')
            self._fig = fig
            self._image = plt.imshow(visual, cmap=cmap, figure=fig)
            plt.ion()
        if plt.fignum_exists('env'):
            # If you close the window, the episode will terminate.
            self._image.set_data(visual)
            self._fig.canvas.draw_idle()
            plt.pause(1.0 / 60.0)
        else:
            del self._fig
            del self._image
            self.done = True

    def load_hierarchy(self, action_hierarchy):
        state_var_values = {}
        # actions = self.hierarchy['actions']
        actions = action_hierarchy.actions
        # Anything that can be passed as a parameter contributes to the number of grounded actions
        # To save memory, only enumerate values for state variables mentioned in params
        possible_params = set()
        '''
        for action in actions:
            if not actions[action]['primitive']:
                for param in actions[action]['params']:
                    possible_params.update(actions[action]['params'][param])
                    '''
        for edge_variables in action_hierarchy.edges.values():
            for variable in edge_variables:
                possible_params.add(variable)

        bound_variables_map = {}

        # Enumerate the values of each state variable if they appear in parameters and create strings
        if 'Orange' in possible_params:
            locations = self.ok_possible_loc + self.ol_possible_loc
            state_var_values['Orange'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Orange_L' in possible_params:
            # locations = self.ol_possible_loc
            locations = [[1,2]]
            state_var_values['Orange_L'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Orange_K' in possible_params:
            locations = self.ok_possible_loc
            state_var_values['Orange_K'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Blue' in possible_params:
            locations = self.bk_possible_loc + self.bl_possible_loc
            state_var_values['Blue'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Blue_L' in possible_params:
            # locations = self.bl_possible_loc
            locations = [[2, 2]]
            state_var_values['Blue_L'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Blue_K' in possible_params:
            locations = self.bk_possible_loc
            state_var_values['Blue_K'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Green' in possible_params:
            locations = self.gk_possible_loc + self.gl_possible_loc
            state_var_values['Green'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Green_L' in possible_params:
            # locations = self.gl_possible_loc
            locations = [[1, 3]]
            state_var_values['Green_L'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Green_K' in possible_params:
            locations = self.gk_possible_loc
            state_var_values['Green_K'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'Gem' in possible_params:
            locations = self.gem_possible_loc
            state_var_values['Gem'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                 bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

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

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[int]:
        return [0]

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list
