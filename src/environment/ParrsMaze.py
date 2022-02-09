"""
Parr's Maze
-1 reward at every time step
Goal is to successfully complete the maze in as few moves as possible
"""
from typing import Dict, List, Tuple, TYPE_CHECKING, Union

from matplotlib.colors import ListedColormap

from common.aux_env_info import AuxiliaryEnvInfo
from common.domain_transfer import DomainTransferMessage
from domain import DiscreteActionDomain
from domain.actions import DiscreteAction
from domain.features import CoordinateFeature, DiscreteFeature, BinaryFeature
from domain.hierarchical_domain import HierarchicalActionDomain, action_hierarchy_from_config
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from environment.actionhierarchy.validate_hierarchy import validate_hierarchy
from common.properties import Properties
from config.config import ConfigItemDesc
from environment import Environment

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from config import Config


def _valid_maze_tiles(data):
    assert isinstance(data, dict), 'The maze tile data must be a dictionary.'
    for k, v in data.items():
        if int(k) == 0:
            assert v == 'EMPTY', 'Type 0 must always be EMPTY'
        elif int(k) == 1:
            assert v == 'WALL', 'Type 1 must always be WALL'
        else:
            return False
    return True


def _valid_room_templates(data):
    assert isinstance(data, list), 'There must be a list of room templates'
    msg1 = 'The room templates must be a list of integer strings.'
    width = None
    for template in data:
        assert isinstance(template, list), msg1
        for row in template:
            assert isinstance(row, str), msg1
            if width is None:
                width = len(row)
            else:
                assert len(row) == width, 'The room templates must be all be the same size, but the row ' \
                                          '[{}] did not match the previous width of [{}]'.format(row, width)
    return True


def _valid_room_map(data):
    msg1 = 'The room map must be a list of lists of integers'
    assert isinstance(data, list), msg1
    for row in data:
        for e in row:
            assert isinstance(e, int), msg1
    return True


class ParrsMaze(Environment):
    """
    Parrs Maze Envrionment
    """
    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc(name="tile_types",
                           check=_valid_maze_tiles,
                           info='Descriptions of the maze tile types'),
            ConfigItemDesc(name='room_templates',
                           check=_valid_room_templates,
                           info='Room templates: the map of empty spaces and walls for a generic room'),
            ConfigItemDesc(name='room_map',
                           check=_valid_room_map,
                           info='Room map: the layout of all rooms and their numbers'),
            ConfigItemDesc(name='room_orientations',
                           check=lambda l: isinstance(l, dict),
                           info='Room orientations: the template and orientation of each numbered room'),
            ConfigItemDesc(name='intersections',
                           check=lambda l: isinstance(l, dict),
                           info='The list of room numbers that represent intersections'),
            ConfigItemDesc(name='starting_room',
                           check=lambda i: isinstance(i, int),
                           info='The room that the agent initially spawns in must be specified'),
            ConfigItemDesc(name='agent_spawn',
                           check=lambda l: isinstance(l, list) and all([isinstance(coord, int)
                                                                        and coord >= 0 for coord in l]),
                           info='Agent spawn coordinates as a duple of ints'),
            ConfigItemDesc(name='default_reward',
                           check=lambda i: isinstance(i, int),
                           info='Default reward for steps elapsed in Taxi World. Typically -1.'),
            ConfigItemDesc(name="action_hierarchy",
                           check=validate_hierarchy,
                           info='The hierarchy of actions that can be executed by the agent',
                           nestable=True),
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def __init__(self, config: 'Config'):

        config = config.find_config_for_instance(self)
        self.config = config

        # Read config vars into instance vars
        self.room_map = np.array(config.room_map)
        self.intersections = config.intersections
        self.tile_dict = config.tile_types
        self.agent_spawn = config.agent_spawn
        self.starting_room = config.starting_room
        self.default_reward = config.default_reward
        self.agent_class = 0

        # Check if the agent system is flat or hierarchical
        self._flat = True
        hierarchy_config_dict = config.action_hierarchy
        if hierarchy_config_dict:
            action_hierarchy = action_hierarchy_from_config(hierarchy_config_dict)
            self._flat = False

        self.wall = "WALL"
        self.empty = "EMPTY"
        self.empty_int = 0
        self.wall_int = 1
        self.rows = len(config.room_templates[0])
        self.cols = len(config.room_templates[0][0])

        self.move_directions = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
        self.directions = [0, 1, 2, 3]
        # TODO are values like [1, 1, 1, 1] really possible?
        self.possible_surroundings = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                                      [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                                      [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                                      [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]

        # Initialize the agent
        self._agent_id_list = [0]
        self._agent_class_map = {0: self.agent_class}

        # Convert agent_spawn to dictionary
        self._agent_positions: Dict[int, List[int, int]] = {0: self.agent_spawn}
        self.agent_loc = self.agent_spawn
        self.agent_room = self.starting_room
        self.agent_direction = -1
        self.left_room = False

        # Save the coordinates of each room in the room map
        self.room_coordinates = {}
        for y, row in enumerate(self.room_map):
            for x, room in enumerate(row):
                if room != -1:
                    self.room_coordinates[room] = [y, x]

        # Convert each room template into a numpy array
        self.room_templates = []
        for template in config.room_templates:
            template_array = np.zeros((self.rows, self.cols))
            for y, row in enumerate(template):
                for x, item in enumerate(row):
                    template_array[y, x] = int(item)
            self.room_templates.append(template_array)

        # Create a dictionary mapping each room number to its layout as a numpy array
        self.room_maps = {}
        for room, orientation in config.room_orientations.items():
            template_num, rotation = orientation[0], orientation[1]
            room_map = np.copy(self.room_templates[template_num])
            self.room_maps[int(room)] = np.rot90(room_map, rotation)
        # Add a room of all walls to create the borders
        self.room_maps[-1] = np.full((self.rows, self.cols), 1)

        # For visualize(), create an np array of the entire map with all rooms
        # Create each row of the full map, one room at a time
        row_list = []
        for row in self.room_map:
            temp_row = self.room_maps[row[0]]
            for i in range(1, len(self.room_map[0])):
                temp_row = np.concatenate((temp_row, self.room_maps[row[i]]), axis=1)
            row_list.append(temp_row)

        # Concatenate each row of the full map
        self.full_map = row_list[0]
        for i in range(1, len(self.room_map)):
            self.full_map = np.concatenate((self.full_map, row_list[i]))

        # If applicable, calculate the number of values each state variable in the hierarchy can take on
        # This is used for grounding parameterized actions in the HierarchicalActionDomain
        if not self._flat:
            state_var_values = {}
            actions = action_hierarchy.actions
            # Anything that can be passed as a parameter contributes to the number of grounded actions
            # To save memory, only enumerate values for state variables mentioned in params
            possible_params = set()

            for edge_variables in action_hierarchy.edges.values():
                for parameter, state_variable in edge_variables.items():
                    possible_params.add(state_variable)

            bound_variables_map = {}

            # Enumerate the values of each state variable if they appear in parameters and create strings
            # Create a reverse mapping of the name to the value for each bound variable
            if 'agent_loc' in possible_params:
                locations = [(y, x) for y in range(self.rows) for x in range(self.cols)]
                sv_values = []
                for l in locations:
                    name = f'_al_{l[0]}_{l[1]}_'
                    sv_values.append(name)
                    bound_variables_map[name] = l
                state_var_values['agent_loc'] = sv_values

            if 'room_num' in possible_params:
                sv_values =[]
                for r in range(-2, np.amax(self.room_map) + 1):
                    name = f'_rn_{r}'
                    sv_values.append(name)
                    bound_variables_map[name] = r
                state_var_values['room_num'] = sv_values

            if 'direction' in possible_params:
                sv_values = []
                for d in self.directions:
                    name = f'_d_{d}'
                    sv_values.append(name)
                    bound_variables_map[name] = d
                state_var_values['direction'] = sv_values

            if 'in_intersection' in possible_params:
                state_var_values['in_intersection'] = [f'_ii_{i}' for i in [0, 1]]
                bound_variables_map['_ii_0'] = 0
                bound_variables_map['_ii_1'] = 1

            if 'left_room' in possible_params:
                state_var_values['left_room'] = [f'_lr_{i}' for i in [0, 1]]
                bound_variables_map['_lr_0'] = 0
                bound_variables_map['_lr_1'] = 1

            if 'surroundings' in possible_params:
                sv_values = []
                for l in self.possible_surroundings:
                    name = f'_s_{l[0]}_{l[1]}_{l[2]}_{l[3]}'
                    sv_values.append(name)
                    bound_variables_map[name] = l
                state_var_values['surroundings'] = sv_values

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

            # Pass the updated hierarchy dictionary to the constructor of the HierarchicalActionDomain
            # Expand action hierarchy
            self.hierarchy = action_hierarchy.compile(state_var_values, bound_variables_map)
            self.hierarchical_action_domain = HierarchicalActionDomain(self.hierarchy.root, self.hierarchy)
            self.hierarchical_observation_domain = self.abstract_all_observation_domains()

        config.seed = np.random.seed()
        super(ParrsMaze, self).__init__(config)

    def get_auxiliary_info(self) -> AuxiliaryEnvInfo:
        if not self._flat:
            assert self.hierarchy, "Cannot get aux info for non-hierarchical learning, joint action domains not supported"
            return AuxiliaryEnvInfo(joint_observation_domains=self.hierarchical_observation_domain,
                                    hierarchy=self.hierarchical_action_domain)

    def set_initial_seed(self, seed: int):
        np.random.seed(seed)

    def get_seed_state(self):
        return np.random.get_state()

    def set_seed_state(self, seed_state):
        np.random.set_state(seed_state)

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        """
        Reset the position of the agent in the maze
        :return: Initial state
        """
        # Reset the initial taxi positions
        self._agent_positions: Dict[int, List[int, int]] = {0: self.agent_spawn}
        self.agent_loc = self.agent_spawn
        self.agent_room = self.starting_room
        self.agent_direction = -1
        self.left_room = False

        return self.agent_room

    def get_tile_in_direction(self, direction: int) -> int:
        move = self.move_directions[direction]
        current_position = self.agent_loc
        target_position = [current_position[0] + move[0], current_position[1] + move[1]]

        # If the target position is in the same room, return the tile
        if target_position[0] in range(self.rows) and target_position[1] in range(self.cols):
            return self.room_maps[self.agent_room][target_position[0]][target_position[1]]

        # Otherwise, this direction is in another room
        # Wrap the position coordinates to the correct values for the next room
        target_position = [target_position[0] % self.rows, target_position[1] % self.cols]

        # Find the coordinates of the next room in the room map
        room_coordinate = self.room_coordinates[self.agent_room]
        target_room = [room_coordinate[0] + move[0], room_coordinate[1] + move[1]]

        # If the target room is out of bounds, return 1 as a wall
        if target_room[0] not in range(len(self.room_map)) or target_room[1] not in range(len(self.room_map[0])):
            return 1

        # Return the tile in the target room
        return self.room_maps[self.room_map[target_room[0]][target_room[1]]][target_position[0]][target_position][1]

    def observe(self, obs_groups=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        """
        Get the current values for each of the state variables
        :return:
        """
        observation = self._observation_domain.generate_empty_array()

        loc_slice = self._observation_domain.index_for_name('agent_loc')
        observation[loc_slice] = [self.agent_loc[0], self.agent_loc[1]]

        room_slice = self._observation_domain.index_for_name('room_num')
        observation[room_slice] = [self.agent_room]

        dir_slice = self._observation_domain.index_for_name('direction')
        observation[dir_slice] = [self.agent_direction]

        inter_slice = self._observation_domain.index_for_name('in_intersection')
        observation[inter_slice] = [int(self.agent_room in self.intersections)]

        left_slice = self._observation_domain.index_for_name('left_room')
        observation[left_slice] = [int(self.left_room)]

        surroundings = [self.get_tile_in_direction(direction) for direction in self.directions]
        surr_slice = self._observation_domain.index_for_name('surroundings')
        observation[surr_slice] = surroundings

        return {0: observation}

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """
        action_index = self.action_domain[self.agent_class].index_for_name('move_actions')

        for agent in actions:
            # All actions are move actions, find which direction the move is
            action = actions[agent][action_index][0]

            # Keep the action the same with probability 0.8, turn right with 0.1, turn left with 0.1
            modification = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
            direction = (action + modification) % 4
            self.agent_direction = direction
            self.left_room = False

            # Calculate the target location of the move
            move = self.move_directions[direction]
            current_position = self.agent_loc
            target_position = [current_position[0] + move[0], current_position[1] + move[1]]

            # If the target position is in the same room, simply update the position
            if target_position[0] in range(self.rows) and target_position[1] in range(self.cols):
                # Move only if the tile is not a wall
                if self.room_maps[self.agent_room][target_position[0]][target_position[1]] == 0:
                    self.agent_loc = target_position
                continue

            # Otherwise, this direction is in another room
            # Wrap the position coordinates to the correct values for the next room
            target_position = [target_position[0] % self.rows, target_position[1] % self.cols]

            # Find the coordinates of the next room in the room map
            room_coordinate = self.room_coordinates[self.agent_room]
            target_room_coords = [room_coordinate[0] + move[0], room_coordinate[1] + move[1]]

            # If the target room is out of bounds, do not move
            if target_room_coords[0] not in range(len(self.room_map)) or target_room_coords[1] not in range(len(self.room_map[0])):
                continue

            # If the target tile in the next room is a wall, do not move
            if self.room_maps[self.room_map[target_room_coords[0]][target_room_coords[1]]][target_position[0]][target_position][1] == 1:
                continue

            # Otherwise, update the appropriate vars
            self.agent_room = self.room_map[target_room_coords[0]][target_room_coords[1]]
            self.agent_loc = target_position
            self.left_room = True

        if self.agent_room == -2:
            self.done = True

        # There are no special rewards, completing the maze in the fewest moves possible minimizes the negative reward
        return {0: self.default_reward}

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[int]:
        return [self.agent_class]

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    def _create_observation_domains(self, config, state_variables=None) -> Dict[int, ObservationDomain]:
        """
        Observation domain contains the state variables for the maze:
         "agent_loc", "room_num", "direction", "in_intersection", "left_room", "surroundings"
        :param config:
        :param state_variables: optional - the observation domain is restricted to only the passed vars
        :return: the observation domain
        """
        # Check if a specific list of state_variables was passed
        # If not, fill with all the state variables
        if state_variables is None:
            state_variables = ['agent_loc', 'room_num', 'direction', 'in_intersection', 'left_room', 'surroundings']

        items = []
        # The agent's position
        if 'agent_loc' in state_variables:
            agent_loc = CoordinateFeature(name='agent_loc', lower=[0, 0],
                                          upper=[self.rows, self.cols],
                                          is_discrete=True)
            items.extend([agent_loc])

        # The current room of the agent
        if 'room_num' in state_variables:
            room_num = DiscreteFeature(name='room_num', size=np.amax(self.room_map) + 3, starts_from=-2)
            items.extend([room_num])

        # The last direction travelled by the agent
        if 'direction' in state_variables:
            direction = DiscreteFeature(name='direction', size=4)
            items.extend([direction])

        # Binary feature indicating if the agent is in an intersection
        if 'in_intersection' in state_variables:
            in_intersection = BinaryFeature(name='in_intersection')
            items.extend([in_intersection])

        # Binary feature indicating if the agent has just changed rooms
        if 'left_room' in state_variables:
            left_room = BinaryFeature(name='left_room')
            items.extend([left_room])

        # List indicating the surroundings of the agent
        if 'surroundings' in state_variables:
            surroundings = CoordinateFeature(name='surroundings',
                                             lower=[0, 0, 0, 0],
                                             upper=[2, 2, 2, 2],  # [1, 1, 1, 1,] < [2, 2, 2, 2]
                                             is_discrete=True,
                                             sparse_values=self.possible_surroundings)
            items.extend([surroundings])

        self._observation_domain = ObservationDomain(items, num_agents=1)
        return {self.agent_class: self._observation_domain}

    def _create_action_domains(self, config) -> Dict[int, ActionDomain]:
        """
        Action domain contains a discrete action item of whether to move North East South or West according to the index
        This environment only has one acting agent
        :param config:
        :return:
        """
        agent_actions = DiscreteAction(name='move_actions', num_actions=4)
        return {self.agent_class: DiscreteActionDomain([agent_actions], 1)}

    def transfer_domain(self, message: DomainTransferMessage) -> DomainTransferMessage:
        pass

    def abstract_all_observation_domains(self):
        h_obs_domain = {}
        for action in self.hierarchy.actions:
            if not self.hierarchy.actions[action]['primitive']:
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.hierarchy.actions[action]['state_variables'])

        return h_obs_domain

    def abstracted_observation_domain(self, state_variables: set) -> ObservationDomain:
        """
        Observation domain contains all the state variables for the maze
        Abstracted domains only hold the state variables that are required at this node in the hierarchy
        :param state_variables: the list of relevant state variables
        :return: Observation Domain
        """
        return self._create_observation_domains(self.config, state_variables=state_variables)[self.agent_class]

    def visualize(self):
        """
        TODO Visualize the current map
        :return:
        """
        # Visualize the entire map (with all rooms)
        visual = np.copy(self.full_map)

        # Calculate the position of the agent relative to the full map
        room_coordinate = self.room_coordinates[self.agent_room]
        row_buffer = self.rows * room_coordinate[0]
        col_buffer = self.cols * room_coordinate[1]
        room_r = self.agent_loc[0]
        room_c = self.agent_loc[1]
        visual[row_buffer + room_r, col_buffer + room_c] = 2
        cmap = ListedColormap(colors=['w', 'k', 'b'])
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
