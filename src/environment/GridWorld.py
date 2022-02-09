"""
grid world
negative reward squares,
decrease reeward at every timestep
goal square
spawn square
walls
terminating tile

ai safety gridworlds

"""
from typing import Dict, List, Tuple, TYPE_CHECKING, Union

from matplotlib.colors import ListedColormap

from domain.DiscreteActionFeatureDomain import DiscreteActionFeatureDomain

from domain.actions import DiscreteAction
from domain.observation import ObservationDomain
from domain.DiscreteActionDomain import DiscreteActionDomain
from domain.ActionDomain import ActionDomain
from domain.features import Feature, DiscreteFeature, OneHotFeature
from common.properties import Properties
from config.config import ConfigItemDesc
from environment import Environment

import numpy as np
import matplotlib.pyplot as plt
if TYPE_CHECKING:
    from config import Config


# TODO add config to prohibit agents from sharing spaces.

def _valid_gridworld_tiles(data):
    assert isinstance(data, dict), 'The gridworld tile data must be a dictionary.'
    for k, v in data.items():
        if int(k) == 0:
            assert v == 'EMPTY', 'Type 0 must always be EMPTY'
        elif int(k) == 1:
            assert v == 'WALL', 'Type 1 must always be WALL'
        else:
            err_msg = 'Types other than "0" nad "1" must be special tiles ' \
                      'specifying [reward] int and [terminate] boolean.'
            assert isinstance(v, dict) and 'reward' in v and 'terminate' in v, err_msg
    return True


def _valid_gridworld_map(data):
    msg1 = 'The gridworld map data must be a list of integer strings.'
    assert isinstance(data, list), msg1
    width = None
    for row in data:
        assert isinstance(row, str), msg1
        if width is None:
            width = len(row)
        else:
            assert len(row) == width, 'The gridworld map data must be rectangular but the row ' \
                                      '[{}] did not match the previous width of [{}]'.format(row, width)
    return True


class GridWorld(Environment):
    """
    GridWorld environment.
    The goal of the environment is to move the agent(s) around to reach the squares with highest reward while avoiding ones with negative reward.
    The config can be set to create the map.
    Example config file:
        {
          "name": "GridWorld",
          "tile_types": {
              "0": "EMPTY",
              "1": "WALL",
              "2": {"reward": 10, "terminate": true},
              "3": {"reward": -10, "terminate": true}
          },
          "map": [
            "0002",
            "0313",
            "0100",
            "0000"
          ],
          "agent_spawn": [[3, 0],[3,3]],
          "default_reward": -1,
          "feature_type": "absolute"
        }

    tile_types must have EMPTY as 0 and wALL as 1. This defines these numbers for the map.
    other tile types can be added to the map. Define these in the tile_types dict as well, defining the reward for stepping
    on these tiles, as well as whether that tile will end the environment.

    the map will define and n x m grid using the numbers from tile_types. Each string is a new row, each char is a new tile in the row.
    agent_spawn defines the spawn locations of k agents.
    default_reward is the reward for standing on an empty square.
    feature_type must be absolute or relative and will determine what kind of features will be obtained:
        the local neighborhood, or the absolute entire map

    Gridworld currently allows agents to move through each other.
    Gridworld is a local gridworld where the agent only sees its immediate surroundings.
    """

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc(name="tile_types",
                           check=_valid_gridworld_tiles,
                           info='Descriptions of Gridworld Tile Types.'),
            ConfigItemDesc(name='map',
                           check=_valid_gridworld_map,
                           info='Gridworld map. A list of strings filled with symbols corresponding to tile type.'),
            ConfigItemDesc(name='agent_spawn',
                           check=lambda l: isinstance(l, list) and
                                           all([isinstance(duple, list) and len(duple) == 2 and
                                                all([isinstance(coord, int) and coord >= 0 for coord in duple])
                                                for duple in l]),
                           info='Agent spawning coordinates as duples of int coordinate locations.'),
            ConfigItemDesc(name='default_reward',
                           check=lambda i: isinstance(i, int),
                           info='Default reward for steps elapsed in gridworld. Typically -1.'),
            ConfigItemDesc(name='collide_reward',
                           check=lambda i: isinstance(i, int),
                           info='Default reward for wasting a turn (e.g. bumping into walls). Typically -2.'),
            ConfigItemDesc(name='feature_type',
                           check=lambda s: s in ['relative', 'absolute', 'actionfeature'],
                           info="The type of observation given to agents."
                                "\n\t'absolute' returns a one-hot state vector based on agents' position, perfectly "
                                "encoding the state."
                                "\n\t'relative' returns one-hot state vectors for each tile type in a neighborhood "
                                "around the agent, making the environment a POMDP."
                                "\n\t'actionfeature' returns one-hot state vectors for each projected action, "
                                "allowing the agent to use a model of the world to see into the future. The result is "
                                "something like an ADF where values are ascribed to each state.")
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def __init__(self, config: 'Config'):
        self.agent_class = 0
        self.wall = "WALL"
        self.empty = "EMPTY"
        self.tile_dict = config.environment.tile_types
        self.relative = config.environment.feature_type == 'relative'
        self.actionfeature = config.environment.feature_type == 'actionfeature'
        self.empty_int = 0
        self.wall_int = 1
        for tile_id, tile_info in self.tile_dict.items():
            assert isinstance(int(tile_id), int), 'Tile IDs for GridWorld must be numeric.'
            if tile_info == 'WALL':
                self.wall_int = int(tile_id)
            elif tile_info == 'EMPTY':
                self.empty_int = int(tile_id)
        self.default_reward = config.environment.default_reward
        self.collide_reward = config.environment.collide_reward
        self.actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])

        self.config_map = config.environment.map
        self.config_agent_spawn = config.environment.agent_spawn
        self._reset_state()

        super(GridWorld, self).__init__(config)
    
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
        map_rows : List[str] = self.config_map
        rows = len(map_rows)
        cols = len(map_rows[0])
        self.map = np.zeros((rows, cols))
        # Convert map to np array
        for y, row in enumerate(map_rows):
            for x, item in enumerate(row):
                self.map[y, x] = int(item)
        self.rows = rows
        self.cols = cols
        self.n = len(self.config_agent_spawn)
        self._agent_id_list = list(range(self.n))
        self._agent_class_map = {agent: self.agent_class for agent in self._agent_id_list}
        #convert agent_spawn to dictionary
        self._agent_positions: Dict[int,List[int, int]] = {}
        for agent, position in enumerate(self.config_agent_spawn):
            self._agent_positions[agent] = position
        #return self.map so self.map is referred to as self.state, but technically the state also includes the agent_positions dict and tilesDict
        return self.map

    def observe(self, obs_groups=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        observations = {}

        if self.relative:
            padded_map = np.pad(self.map, [[1, 1], [1, 1]], mode='constant', constant_values=self.wall_int)
        for agent in self.agent_id_list:
            position = self._agent_positions[agent]
            # make observation
            observation = self._observation_domain.generate_empty_array()
            # (r-1, c-1), (r-1, c), (r-1, c+1), (r, c-1), (r, c), (r, c+1), (r+1, c-1), (r+1, c), (r+1, c+1)
            y = position[0]
            x = position[1]

            if self.relative:
                visible = padded_map[y:y+3, x:x+3] if self.relative else self.map
                blocked_slice = self._observation_domain.index_for_name("blocked")
                observation[blocked_slice] = (visible == self.wall_int).flatten() * 2 - 1

                for tile_id, tile_info in self.tile_dict.items():
                    if tile_info not in [self.wall, self.empty]:
                        tile_slice = self._observation_domain.index_for_name("tile_{}".format(tile_id))
                        observation[tile_slice] = (visible == int(tile_id)).flatten() * 2 - 1
            elif self.actionfeature:
                observation = observation.reshape(len(self.actions), *self.map.shape)
                observation[:] = 0
                position = np.array(position)
                for i, action in enumerate(self.actions):
                    proj_position = position + action
                    try:
                        target_tile_type = self.tile_dict[str(int(self.map[proj_position[0], proj_position[1]]))]
                    except IndexError:
                        target_tile_type = self.wall
                    if target_tile_type == self.wall:
                        proj_position = position
                    observation[i, proj_position[0], proj_position[1]] = 1
                observation = observation.ravel()
            else:
                slicer = self._observation_domain.index_for_name('one_hot_position')
                observation[:] = 0
                observation[slicer].reshape(self.map.shape)[y, x] = 1

            # make position observation
            # position_slice = self._observation_domain.index_for_name("agent_position")
            # observation[position_slice] = position
            observations[agent] = observation

        return observations

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """
        rewards = {}
        move_index = self.action_domain[self.agent_class].index_for_name('move')
        for agent in actions:
            move = self.actions[actions[agent][move_index]][0]
            current_position = self._agent_positions[agent]
            target_position = current_position + move
            if target_position[0] not in range(self.rows) or target_position[1] not in range(self.cols):
                target_position = current_position #that tile is outside of our range

            target_tile_type = self.tile_dict[str(int(self.map[target_position[0], target_position[1]]))]

            #check what tile it is moving to. the only tile we can't move to are WALL.
            # for simplicity i will say agents can move on top of eachother, since I'm not actually putting the agents
            # on the map, and just keeping their position in the _agent_positions dict
            if target_tile_type == self.wall:
                target_position = current_position
                target_tile_type = self.tile_dict[str(int(self.map[target_position[0], target_position[1]]))]

            if not np.array_equal(target_position, self._agent_positions[agent]):
                if isinstance(target_tile_type, dict) and target_tile_type['terminate']:
                    self.terminated_count += 1

            #move the agent
            self._agent_positions[agent] = target_position

            #calculate reward
            reward = 0
            if np.all(target_position == current_position):
                reward = self.collide_reward
            elif target_tile_type == self.empty:
                reward = self.default_reward
            else:
                reward = target_tile_type["reward"]
                if (not np.array_equal(target_position, current_position)) and target_tile_type["terminate"]:
                    self.terminated_count += 1

            rewards[agent] = reward

        if self.terminated_count >= self.n:
            self.done = True

        return rewards

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
        # agent_position: DomainItem = DomainItem(name='agent_position', shape=[2], dtype='int',
        #                                         drange=slice(0, self.rows))

        # all 9 tiles of this agent, including the one its standing on. in this order:
        # (r-1, c-1), (r-1, c), (r-1, c+1), (r, c-1), (r, c), (r, c+1), (r+1, c-1), (r+1, c), (r+1, c+1)
        # where r,c is the tile the agent is on. If a tile doesn't exist, it will be -1
        # 1 if tile has wall, -1 otherwise
        if self.relative:
            obs_shape = 9
            blocked: Feature = Feature(name='blocked', shape=[obs_shape], dtype='int', drange=slice(-1, 2))
            special_tiles: List[Feature] = []
            for tile_id, tile_type in self.tile_dict.items():
                if tile_type not in [self.wall, self.empty]:
                    special_tiles.append(Feature(name='tile_{}'.format(tile_id), shape=[obs_shape],
                                                 dtype='int', drange=slice(-1, 2)))
            domain = ObservationDomain([blocked] + special_tiles, num_agents=self.n)
        else:
            pos = OneHotFeature('one_hot_position', int(np.prod(self.map.shape)))
            domain = ObservationDomain([pos], num_agents=self.n)
            if self.actionfeature:
                action_domain = self.agent_class_action_domains[self.agent_class]
                domain = DiscreteActionFeatureDomain(domain, action_domain)
        self._observation_domain = domain
        return {self.agent_class: domain}

    def _create_action_domains(self, config) -> Dict[int, ActionDomain]:
        '''
        Action domain is a single discrete action item of whether to move Up, Down, Left, Right according to the index
        :param config:
        :return:
        '''
        agent_action = DiscreteAction(name='move', num_actions=len(self.actions))
        return {self.agent_class: DiscreteActionDomain([agent_action], 1)}

    def visualize(self):
        """
        Visualize the current map. Going to try it first with matplotlib
        :return:
        """
        # TODO (future) Switch to pyglet visualization.
        visual = np.copy(self.map)
        map = self.map
        for agent in self._agent_positions:
            r = self._agent_positions[agent][0]
            c = self._agent_positions[agent][1]
            visual[r,c] = 5
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
            plt.pause(1.0/60.0)
        else:
            del self._fig
            del self._image
            self.done = True
        # plt.colorbar()

