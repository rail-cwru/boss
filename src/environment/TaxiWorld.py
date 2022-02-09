"""
Taxi World
-1 reward at every time step
Reward for dropping off passenger at destination - termination condition
Passenger randomly spawns in one of four locations
Walls

"""
import random
from typing import Dict, List, Tuple, TYPE_CHECKING, Union
from matplotlib.colors import ListedColormap
from common.aux_env_info import AuxiliaryEnvInfo
from common.domain_transfer import DomainTransferMessage
from domain import DiscreteActionDomain
from domain.actions import DiscreteAction
from domain.features import CoordinateFeature, VectorFeature, DiscreteFeature, Feature
from domain.hierarchical_domain import HierarchicalActionDomain, action_hierarchy_from_config
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain

from common.properties import Properties
from config.config import ConfigItemDesc
from environment.HierarchicalEnvironment import HierarchicalEnvironment
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from config import Config


def _valid_taxiworld_tiles(data):
    assert isinstance(data, dict), 'The Taxi World tile data must be a dictionary.'
    for k, v in data.items():
        if int(k) == 0:
            assert v == 'EMPTY', 'Type 0 must always be EMPTY'
        elif int(k) == 1:
            assert v == 'WALL', 'Type 1 must always be WALL'
        elif int(k) == 2:
            assert v == 'S/D', 'Type 2 must always be a passenger source or destination (S/D)'
        else:
            # TODO remove references to special tiles? Pseudo-rewards will not be implemented here
            err_msg = 'Types other than "0", "1", and "2" must be special tiles ' \
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


def _valid_action_hierarchy(ah):
    # TODO
    return True


class TaxiWorld(HierarchicalEnvironment):
    """
    Taxi World environment.
    The goal of the environment is to move the taxi to pick up a passenger and bring them to their desired destination.
    The config can be set to create the map.
    Example config file:
    "name": "TaxiWorld",
    "action_hierarchy": "./actionhierarchy/taxiworld.json",
    "tile_types": {
        "0": "EMPTY",
        "1": "WALL",
        "2": "S/D"
    },
    "map": [
        "20010002",
        "00010000",
        "00000000",
        "01000100",
        "21000102"
    ],
    "agent_spawn": [[3,3],[3,4]],
    "num_passengers": 2,
    "default_reward": -1,
    "feature_type": "absolute"

    action_hierarchy defines the location of the specified hierarchy

    tile_types must have EMPTY as 0 and WALL as 1. The corners are type 2 and define the initial spawnpoints of the
    passenger(s). Other tile types can be added to the map. Define these in the tile_types dict as well, specifying the
    reward for stepping on these tiles, as well as whether that tile will end the environment.

    map will define a nxm grid using the numbers from tile_types. Each string is a new row, each char is a new
    tile in the row.

    agent_spawn defines the spawn location(s) of the Taxi(s).

    default_reward is the reward for one action.

    feature_type must be absolute or relative and will determine what kind of features will be obtained:
        the local neighborhood, or the absolute entire map
    """

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
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
            ConfigItemDesc(name='feature_type',
                           check=lambda s: s in ['absolute'],
                           info="The type of observation given to agents."
                                "\n\t'absolute' returns a one-hot state vector based on agents' position, perfectly "
                                "encoding the state."
                                "\n\t'relative' returns one-hot state vectors for each tile type in a neighborhood "
                                "around the agent, making the environment a POMDP."),
            ConfigItemDesc(name="num_passengers",
                           check=lambda i: isinstance(i, int),
                           info='The number of passengers to be spawned in the world'),
            ConfigItemDesc(name="action_hierarchy",
                           check=_valid_action_hierarchy,
                           info='The hierarchy of actions that can be executed by the agent',
                           nestable=True),
            ConfigItemDesc(name="stochastic",
                           check=lambda i: isinstance(i, bool),
                           info='The environment can slightly stochastic')
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def __init__(self, config: 'Config'):

        config = config.find_config_for_instance(self)
        # assert config.domain == HierarchicalActionDomain

        self.config_map = config.map
        self.config_agent_spawn = config.agent_spawn
        self.num_passengers = config.num_passengers
        self.tile_dict = config.tile_types
        self.default_reward = config.default_reward
        self.stochastic = config.stochastic
        self.reset_dest = False
        self.agent_class = 0

        self.wall = "WALL"
        self.empty = "EMPTY"
        self.sd = "S/D"
        self.empty_int = 0
        self.wall_int = 1
        self.sd_int = 2
        self._set_environment_init_state()
        for tile_id, tile_info in self.tile_dict.items():
            assert isinstance(int(tile_id), int), 'Tile IDs for GridWorld must be numeric.'
            if tile_info == 'WALL':
                self.wall_int = int(tile_id)
            elif tile_info == 'EMPTY':
                self.empty_int = int(tile_id)
            elif tile_info == 'S/D':
                # self.empty_int = int(tile_id)
                self.sd_int = int(tile_id)

        self.move_actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        config.seed = 2
        np.random.seed()
        super(TaxiWorld, self).__init__(config)

        self.reward_range = 30

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

    def load_hierarchy(self, action_hierarchy):
        state_var_values = {}
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
        if 'taxi_loc' in possible_params:
            locations = np.argwhere(self.map != 1).tolist()
            state_var_values['taxi_loc'] = [f'_{l[0]}_{l[1]}_' for l in locations]

        if 'source' in possible_params:
            state_var_values['source'] = [f'_{l[0]}_{l[1]}_' for l in self._sd_locations]
            for l in self._sd_locations:
                bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'destination' in possible_params:
            state_var_values['destination'] = [f'_{l[0]}_{l[1]}_' for l in self._sd_locations]
            for l in self._sd_locations:
                bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'holding_passenger' in possible_params:
            # This state variable will never be used as a parameter since it is the only one of its type
            raise NotImplementedError()

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
        self.terminated_count = 0

        # Convert String-representation of map to np array
        map_rows: List[str] = self.config_map
        self.rows = len(map_rows)
        self.cols = len(map_rows[0])
        self.map = np.zeros((self.rows, self.cols))
        for y, row in enumerate(map_rows):
            for x, item in enumerate(row):
                self.map[y, x] = int(item)

        # Initialize each Taxi
        self.n = len(self.config_agent_spawn)
        self._agent_id_list = list(range(self.n))
        self._agent_class_map = {agent: self.agent_class for agent in self._agent_id_list}

        # Convert agent_spawn to dictionary
        self._agent_positions: Dict[int, List[int, int]] = {}
        for agent, position in enumerate(self.config_agent_spawn):
            if position == 'random':
                self.set_rand_spawn(agent)
            else:
                self._agent_positions[agent] = position

        # Initialize the passengers
        self._sd_locations = np.argwhere(self.map == 2).tolist()
        self._passenger_sources: Dict[int, List[int, int]] = {}
        self._passenger_destinations: Dict[int, List[int, int]] = {}
        self._passenger_picked_up: Dict[int, int] = {}

        for i in range(self.num_passengers):
            # Assign each passenger a random source/spawn location (may overlap)
            self._passenger_sources[i] = random.choice(self._sd_locations)

            # Assign each passenger a random destination that is different from their source
            destinations = [d for d in self._sd_locations if d != self._passenger_sources[i]]
            self._passenger_destinations[i] = random.choice(destinations)

            self._passenger_picked_up[i] = -1

        return self.map

    def set_rand_spawn(self, agent):
        is_wall = True
        while is_wall:
            row = np.random.choice(range(0,self.rows))
            col = np.random.choice(range(0,self.cols))
            target_tile_type = self.tile_dict[str(int(self.map[row, col]))]
            if target_tile_type != self.wall and target_tile_type != self.sd:
                position = [row, col]
                if position not in self._agent_positions.values():
                    is_wall = False
        self._agent_positions[agent] = position

    def set_spawn(self, pos):
        self._agent_positions[0] = pos
        self._passenger_sources[0] = [0, 0]
        self._passenger_destinations[0] = [4, 4]

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

        # The taxi numbers and class map do not need to be reset
        # Reset the initial taxi positions
        self._agent_positions: Dict[int, List[int, int]] = {}
        for agent, position in enumerate(self.config_agent_spawn):
            if position == 'random':
                self.set_rand_spawn(agent)
            else:
                self._agent_positions[agent] = position

        # Initialize the passengers
        self._passenger_sources: Dict[int, List[int, int]] = {}
        self._passenger_destinations: Dict[int, List[int, int]] = {}
        self._passenger_picked_up: Dict[int, int] = {}
        for i in range(self.num_passengers):
            # Assign each passenger a random source/spawn location (may overlap)
            self._passenger_sources[i] = random.choice(self._sd_locations)

            # Assign each passenger a random destination that is different from their source
            destinations = [d for d in self._sd_locations if d != self._passenger_sources[i]]
            self._passenger_destinations[i] = random.choice(destinations)

            # Indicate that none of the passengers have been picked up
            self._passenger_picked_up[i] = -1

        return self.map

    def observe(self, obs_groups=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        """
        :return:
        """
        observations = {}

        # Other than the taxi location, every agent gets the same observation
        pre_observation = self._observation_domain.generate_empty_array()

        for p, source in self._passenger_sources.items():
            p_source_slice = self._observation_domain.index_for_name(name=f'{p}', prefix='source')
            pre_observation[p_source_slice] = [source[0], source[1]]

        for p, dest in self._passenger_destinations.items():
            p_dest_slice = self._observation_domain.index_for_name(name=f'{p}', prefix='destination')
            pre_observation[p_dest_slice] = [dest[0], dest[1]]

        for p, picked_up in self._passenger_picked_up.items():
            pass_picked_up_slice = self._observation_domain.index_for_name(name=f'{p}', prefix='holding_passenger')
            pre_observation[pass_picked_up_slice] = [picked_up]

        for agent in self.agent_id_list:
            observation = pre_observation.copy()
            taxi_loc_slice = self._observation_domain.index_for_name(name='taxi_loc')

            pos = self._agent_positions[agent]

            observation[taxi_loc_slice] = [pos[0], pos[1]]
            observations[agent] = observation

        return observations

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """
        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('taxi_actions')
        for agent in actions:
            action = actions[agent][action_index][0]
            reward = self.default_reward

            # If there is an extra set of NSWE actions
            if action > 5:
                action -= 6

            if action < 4:

                if self.stochastic:
                    choice1 = action - 1 if action - 1 >= 0 else 3
                    choice2 = action + 1 if action + 1 < 4 else 0
                    action = np.random.choice([choice1, action, choice2], 1, p=[0.1, 0.8, 0.1])[0]

                # move action
                move = self.move_actions[action]
                current_position = self._agent_positions[agent]
                # TODO check for y,x
                target_position = [current_position[0] + move[0], current_position[1] + move[1]]

                # Check that the target position is inside the bounds of the map
                if target_position[0] not in range(self.rows) or target_position[1] not in range(self.cols):
                    # Outside of map, don't move
                    target_position = current_position

                # Check what tile it is moving to. the only tile we can't move to are WALL.
                target_tile_type = self.tile_dict[str(int(self.map[target_position[0], target_position[1]]))]

                if target_tile_type == self.wall:
                    # Is a wall, don't move
                    target_position = current_position

                # Move the agent
                self._agent_positions[agent] = target_position

            elif action == 4:
                # pickup action
                current_position = self._agent_positions[agent]
                # since several passengers could spawn in the same position, find the first one that is not picked up
                for p, loc in self._passenger_sources.items():
                    if loc == current_position and self._passenger_picked_up[p] == -1:
                        self._passenger_picked_up[p] = agent
                        self.reset_dest = True
                        break
                else:
                    # all of the passengers in this location have been picked up
                    # -10 reward for illegally trying to pickup
                    reward += -10

            elif action == 5:
                # putdown action
                current_position = self._agent_positions[agent]
                for p, loc in self._passenger_destinations.items():
                    # check that this is the location that the passenger wants to be in
                    if loc == current_position and self._passenger_picked_up[p] == agent:
                        self.terminated_count += 1
                        # -2 ... -n - 1 are agents that held the passenger before being dropped off
                        self._passenger_picked_up[p] = -2 - agent
                        reward += 20
                        break
                else:
                    # The taxi is not holding any passengers
                    # -10 reward for illegally trying to pickup
                    reward += -10

            rewards[agent] = reward

        if self.terminated_count >= self.num_passengers:
            self.done = True

        return rewards

    def reset_destination(self, agent):
        destinations = [d for d in self._sd_locations if
                        (d != self._passenger_sources[agent] and d != self._passenger_destinations[agent])]
        destinations.append(self._passenger_destinations[agent])
        dest_range = range(0, len(destinations))
        index = np.random.choice(dest_range, 1, p=[.15, .15, .7])[0]
        self._passenger_destinations[agent] = destinations[index]

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[int]:
        return [0]

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

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
        taxi_loc = CoordinateFeature(name='taxi_loc', lower=[0, 0],
                                     upper=[self.rows, self.cols],
                                     is_discrete=True)
        items.extend([taxi_loc])

        # The list of passenger sources
        for p in range(self.num_passengers):
            # TODO fix type hinting here: https://stackoverflow.com/questions/53974936/pycharm-and-type-hinting-warning
            p_loc = CoordinateFeature(name=f'{p}', lower=[0, 0],
                                      upper=[self.rows, self.cols],
                                      is_discrete=True, sparse_values=self._sd_locations,
                                      prefix='source')

            items.extend([p_loc])

        # The list of passenger destinations
        for p in range(self.num_passengers):
            p_loc = CoordinateFeature(name=f'{p}', lower=[0, 0],
                                      upper=[self.rows, self.cols],
                                      is_discrete=True, sparse_values=self._sd_locations,
                                      prefix='destination')

            items.extend([p_loc])

        # The list of passenger statuses
        for p in range(self.num_passengers):
            # if we had multiple taxis we coudl make 0,1,2... to show which taxi
            holding_passenger = DiscreteFeature(name=f'{p}', size= 2 * self.n + 1, starts_from= -1 * self.n - 1,
                                                prefix='holding_passenger')

            items.append(holding_passenger)

        self._observation_domain = ObservationDomain(items, num_agents=self.n)
        return {self.agent_class: self._observation_domain}

    def _create_action_domains(self, config) -> Dict[int, ActionDomain]:
        """
        Action domain contains a discrete action item of whether to move North East South or West according to the index
        it also contains two actions to interact with the passenger: 'pickup' and 'putdown'
        :param config:
        :return:
        """
        taxi_actions = DiscreteAction(name='taxi_actions', num_actions=6)
        return {self.agent_class: DiscreteActionDomain([taxi_actions], self.n)}

    def transfer_domain(self, message: DomainTransferMessage) -> DomainTransferMessage:
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

    def abstracted_observation_domain(self, state_variables: set) -> ObservationDomain:
        """
        Observation domain contains the state variables for Taxi World
        Every taxi can see every other taxi, passenger, and passenger status
        Holds only the state variables that are required at this node in the hierarchy
        :return: Observation Domain
        """
        items = []

        if 'taxi_loc' in state_variables:
            # The agent's position
            taxi_loc = CoordinateFeature(name='taxi_loc', lower=[0,0],
                                         upper=[self.rows, self.cols],
                                         is_discrete=True)
            items.extend([taxi_loc])

        if 'source' in state_variables:
            # The list of passenger sources
            for p in range(self.num_passengers):
                p_loc = CoordinateFeature(name=f'{p}', lower=[0,0],
                                          upper=[self.rows, self.cols],
                                          is_discrete=True, sparse_values=self._sd_locations,
                                          prefix='source')
                items.extend([p_loc])

        if 'destination' in state_variables:
            # The list of passenger destinations
            for p in range(self.num_passengers):
                p_loc = CoordinateFeature(name=f'{p}', lower=[0, 0],
                                          upper=[self.rows, self.cols],
                                          is_discrete=True, sparse_values=self._sd_locations,
                                          prefix='destination')

                items.extend([p_loc])

        if 'holding_passenger' in state_variables:
            # The list of passenger statuses
            for p in range(self.num_passengers):
                # if we had multiple taxis we coudl make 0,1,2... to show which taxi
                holding_passenger = DiscreteFeature(name=f'{p}', size=2 * self.n + 1, starts_from=-1 * self.n - 1,
                                                    prefix='holding_passenger')
                items.append(holding_passenger)

        return ObservationDomain(items, num_agents=self.n)
