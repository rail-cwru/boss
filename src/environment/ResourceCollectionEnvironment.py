from functools import partial
from heapq import heappush, heappushpop
from typing import List, Tuple, Dict, Union, TYPE_CHECKING

import numpy as np

from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from domain.features import Feature
from common.properties import Properties
from config.config import ConfigItemDesc
from environment import Environment

if TYPE_CHECKING:
    from config import Config, gen_check, checks

KEY_WALL = -1
TYPE_WORKER = 0
TYPE_TOWNHALL = 1
TYPE_RESOURCE = 2


class ResourceCollectionEnvironment(Environment):
    """
    Single-player resource collection environment.
    The goal is to quickly collect resources.
    Workers gather one of a resource per turn.
    Since the utility of resources can be adjusted, the amount is kept standard.

    The map for this environment is defined by:
        'map_shape'  - A positive duplet of height and width

        Top left is (0, 0).
        'spawn_town_hall' - List of [y, x, can_wood, can_gold] where deposit locations are spawned.
                               can_wood and can_gold should be booleans indicating that the resource may be dropped off
                               at this deposit building. The typical "town hall" allows both.
        'spawn_worker'       - List of duples [y, x] where workers are spawned.
        'spawn_resource'     - List of tuples [y, x, wood, gold] where resource is spawned at [y, x]
                               with amounts [wood, gold]
        'spawn_wall'         - List of duples [y, x] where walls should be spawned.

        'allow_new_worker'   - Boolean. Allows new workers to be built when true.
        'worker_cost'        - Dictionary of {wood: wood_cost, gold: gold_cost}. Describes the cost to build a worker.
        'worker_capacity'       - Dictionary of {wood: wood_limit, gold: gold_limit}. Describes how much a worker can carry

        'wood_reward'        - Reward for delivering one unit of wood.
        'gold_reward'        - Reward for delivering one unit of gold.
        'wait_reward'        - Reward for not delivering anything. Typically will be zero or negative.

        'resources_visible'  - Maximum number of nearest resources a worker will be able to observe.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc('map_shape', partial(gen_check.n_list_func, 2, checks.positive_integer),
                           info='Duple of positive ints describing the 2D map shape rectangle.'),
            ConfigItemDesc('spawn_town_hall',
                           partial(gen_check.nonempty_list_func,
                                   lambda x: len(x) == 4 and isinstance(x[2], bool) and isinstance(x[3], bool)),
                           info='List of [y, x, can_wood, can_gold] descriptions of town halls. '
                                'y and x are the coordinates of the town hall and can_wood and can_gold respectively '
                                'describe whether the town hall accepts depositing wood and gold.'),
            ConfigItemDesc('spawn_worker',
                           partial(gen_check.nonempty_list_func,
                                   partial(gen_check.n_list_func, 2, checks.nonnegative_integer)),
                           info='List of [y, x] coordinates where to spawn worker(s).'),
            ConfigItemDesc('spawn_resource',
                           partial(gen_check.nonempty_list_func,
                                   partial(gen_check.n_list_func, 4, checks.nonnegative_integer)),
                           info='List of [y, x, amt_wood, amt_gold] descriptions of resources, where '
                                'resources are spawned at locations [y, x] with resource amounts [amt_wood, amt_gold]'),
            ConfigItemDesc('spawn_wall',
                           partial(gen_check.nonempty_list_func,
                                   partial(gen_check.n_list_func, 2, checks.nonnegative_integer)),
                           info='List of [y, x] coordinates where to spawn impassable walls.'),
            ConfigItemDesc('allow_new_worker', lambda b: isinstance(b, bool),
                           info='Whether or not to allow townhalls to buy new workers.'),
            ConfigItemDesc('worker_cost',
                           partial(gen_check.uniform_dict, ['wood', 'gold'], checks.positive_integer),
                           info='Dictionary with costs of "wood" and "gold" for creating a worker.'),
            ConfigItemDesc('worker_capacity',
                           partial(gen_check.uniform_dict, ['wood', 'gold'], checks.positive_integer),
                           info='Dictionary with maximum amount of "wood" and "gold" a single worker can carry'),
            ConfigItemDesc('wood_reward', checks.numeric, info='Reward for depositing wood'),
            ConfigItemDesc('gold_reward', checks.numeric, info='Reward for depositing gold'),
            ConfigItemDesc('wait_reward', checks.numeric, info='Reward (usually negative) for passing a turn.'),
            ConfigItemDesc('resources_visible', checks.positive_integer,
                           info='Maximum number of resources any worker can observe.')
        ]

    @classmethod
    @property
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,     # TODO WIP to True
                          use_agent_addition=False,         # TODO WIP to True
                          use_agent_deletion=False,         # TODO WIP to True
                          )

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[str]:
        if self.allow_new_worker:
            return ['worker', 'town_hall']
        else:
            return ['worker']

    def __init__(self, config: 'Config'):
        self.config = config.find_config_for_instance(self)
        self.map_shape = self.config.map_shape
        self.map = np.zeros(self.map_shape)

        self.rows = np.size(self.map, 0)
        self.cols = np.size(self.map, 1)

        self.wall_id = -1

        self.dtype_town_hall = {'dtype': ['int8', 'bool', 'bool', 'int64', 'int64'],
                                'names': ['type', 'accept_wood', 'accept_gold', 'x', 'y']}
        self.dtype_worker = {'dtype': ['int8', 'int64', 'int64', 'int64', 'int64'],
                             'names': ['type', 'amt_wood', 'amt_gold', 'x', 'y']}
        self.dtype_resource = {'dtype': ['int8', 'int64', 'int64', 'int64', 'int64'],
                               'names': ['type', 'amt_wood', 'amt_gold', 'x', 'y']}

        self.allow_new_worker = self.config.allow_new_worker
        self.worker_cost = self.config.worker_cost
        self.worker_capacity = self.config.worker_capacity
        self.wood_reward = self.config.wood_reward
        self.gold_reward = self.config.gold_reward
        self.wait_reward = self.config.wait_reward
        self.resources_visible = self.config.resources_visible

        self.config_spawn_worker = self.config.spawn_worker
        self.config_spawn_town_hall = self.config.spawn_town_hall
        self.config_spawn_resource = self.config.spawn_resource
        self.config_spawn_wall = self.config.spawn_wall

        # map number 0-8 to a direction
        # TODO make into numpy array and find where uses assume it is a dict[int, tuple[int, int]]
        self.direction_map = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 1),
            5: (1, -1),
            6: (1, 0),
            7: (1, 1)
        }

        super().__init__(config)

    # From what I can tell this env does not use randomness
    def set_initial_seed(self, seed: int):
        pass

    def get_seed_state(self):
        return []

    def set_seed_state(self, seed_state):
        pass

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        obj_key = 1
        self.key_obj_map = {}

        agents = []
        agent_class_map = {}

        # Set up workers
        for y, x in self.config_spawn_worker:
            self.map[y, x] = obj_key
            agents.append(obj_key)
            agent_class_map[obj_key] = 0
            self.key_obj_map[obj_key] = np.array([(TYPE_WORKER, 0, 0, x, y)], dtype=self.dtype_worker)
            obj_key += 1

        # Set up townhalls.
        for y, x, can_wood, can_gold in self.config_spawn_town_hall:
            self.map[y, x] = obj_key
            if self.allow_new_worker:
                agents.append(obj_key)
                agent_class_map[obj_key] = 1
            self.key_obj_map[obj_key] = np.array([(TYPE_TOWNHALL, can_wood, can_gold, x, y)],
                                                 dtype=self.dtype_town_hall)
            obj_key += 1

        # Set up resources
        for y, x, amt_wood, amt_gold in self.config_spawn_resource:
            self.map[y, x] = obj_key
            self.key_obj_map[obj_key] = np.array([(TYPE_RESOURCE, amt_wood, amt_gold, x, y)], dtype=self.dtype_resource)
            obj_key += 1

        # Set up walls
        for y, x in self.config_spawn_wall:
            self.map[y, x] = KEY_WALL

        self._agent_id_list = agents
        self._agent_class_map = agent_class_map

        self.current_gold = 0  # amount of resources the player owns
        self.current_wood = 0
        return self.map

    def _create_action_domains(self, config) -> Dict[Union[str, int], ActionDomain]:
        # Townhall, if self.allow_new_worker: 1 Discrete action with range 0, 1
        # 0: Do nothing; 1: Build new worker (spawns randomly)
        action_domains = {}
        num_townhalls = sum(1 for a in self.agent_id_list if self.agent_class_map[a] == 1)
        if self.allow_new_worker:
            build_worker = Feature(name='build_worker', shape=[1], dtype='int', drange=slice(0, 2))
            townhall_action_domain = ActionDomain([build_worker], num_agents=num_townhalls)
            action_domains[1] = townhall_action_domain
        else:
            action_domains[1] = ActionDomain([], num_townhalls)
        # Worker: 1 Discrete Action with range 0, 1, 2, ... 16
        # 8 movement directions, 8 harvest directions, 1 deposit action
        worker_action = Feature(name='worker_action', shape=[1], dtype='int', drange=slice(0, 17))
        num_workers = sum(1 for a in self.agent_id_list if self.agent_class_map[a] == 0)
        action_domains[0] = ActionDomain([worker_action], num_agents=num_workers)

        return action_domains

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        rewards = {}
        townhall_action_domain = self.agent_class_action_domains[1]
        worker_action_domain = self.agent_class_action_domains[0]
        for agent_id in self.agent_id_list:
            if agent_id in actions:
                action = actions[agent_id]
                agent_class = self.agent_class_map[agent_id]
                if agent_class == 1 and self.allow_new_worker:
                    # townhall
                    if action[townhall_action_domain.index_for_name('build_worker')] == 1:
                        # build worker
                        obj_type, accept_wood, accept_gold, x, y = self.key_obj_map[agent_id]
                        # try each adjacent position to find an empty one.
                        for xi in range(x - 1, x + 2):
                            for yi in range(y - 1, y + 2):
                                if xi != x or yi != y:
                                    if 0 <= xi <= np.size(self.map, 1) and 0 <= y <= np.size(self.map, 0) and self.map[
                                        yi, xi] == 0:
                                        # can build worker here
                                        # TODO create new worker at xi yi
                                        pass
                elif agent_class == 0:
                    # worker
                    action_num = action[worker_action_domain.index_for_name('worker_action')]
                    agent_obj = self.key_obj_map[agent_id]
                    obj_type, amt_wood, amt_gold, x, y = agent_obj
                    if amt_wood > 0:
                        resource_type = 0
                    elif amt_gold > 0:
                        resource_type = 1
                    else:
                        resource_type = -1
                    if action_num == 16:
                        if resource_type != -1:
                            for xi in range(x - 1, x + 2):
                                for yi in range(y - 1, y + 2):
                                    if xi != x or yi != y:
                                        if 0 <= xi <= np.size(self.map, 1) and 0 <= y <= np.size(self.map, 0) and \
                                                self.map[yi, xi] != 0:
                                            # check if the agent at xi yi is a townhall
                                            adj_id = self.map[yi, xi]
                                            obj_type = self.key_obj_map[adj_id][0]
                                            if obj_type == TYPE_TOWNHALL:
                                                obj_type, accept_wood, accept_gold, x, y = self.key_obj_map[adj_id]
                                                if resource_type == 0 and accept_wood:
                                                    # deposit wood
                                                    self.current_wood += amt_wood
                                                    amt_wood = 0
                                                    rewards[agent_id] = self.wood_reward

                                                elif resource_type == 1 and accept_gold:
                                                    # deposit gold
                                                    self.current_gold += amt_gold
                                                    amt_gold = 0
                                                    rewards[agent_id] = self.gold_reward
                    elif action_num >= 8:
                        # harvest action
                        # TODO fix me; worker capacity is a dict
                        if amt_wood + amt_gold < self.worker_capacity:
                            # has space to collect more
                            yd, xd = self.direction_map[action_num % 8]
                            yi = y + yd
                            xi = x + xd
                            adj_id = self.map[yi, xi]
                            obj_type = self.key_obj_map[adj_id][0]
                            if obj_type == TYPE_RESOURCE:
                                obj_type, res_wood, res_gold, _, _ = self.key_obj_map[adj_id]
                                if res_wood > 0:
                                    res_contains_type = 0
                                elif res_gold > 0:
                                    res_contains_type = 1
                                else:
                                    res_contains_type = -2  # no resources to collect

                                if resource_type == res_contains_type or resource_type == -1:
                                    # can harvest
                                    if res_contains_type == 0:
                                        # harvest wood
                                        res_wood -= 1
                                        amt_wood += 1
                                    elif res_contains_type == 1:
                                        # harvest gold
                                        res_gold -= 1
                                        amt_gold += 1

                                    if res_wood + res_gold <= 0:
                                        # remove empty resource
                                        del self.key_obj_map[adj_id]
                                        map[yi, xi] = 0
                                    else:
                                        self.key_obj_map[adj_id] = (obj_type, res_wood, res_gold, xi, yi)
                    else:
                        # move action
                        yd, xd = self.direction_map[action_num % 8]
                        yi = y + yd
                        xi = x + xd
                        if self.map[yi, xi] == 0:
                            # can move here
                            self.map[yi, xi] = agent_id
                            self.map[y, x] = 0
                            y = yi
                            x = xi

                    self.key_obj_map[agent_id] = (obj_type, amt_wood, amt_gold, x, y)

            if agent_id not in rewards:
                rewards[agent_id] = self.wait_reward

        return rewards

    def _create_observation_domains(self, config) -> Dict[Union[str, int], ObservationDomain]:
        # Townhall, if self.allow_new_worker:
        # [amt_workers, amt_wood, amt_gold, occupied_neighbors]
        amt_workers = Feature('amt_workers', shape=[1], dtype='int',
                              drange=slice(0, None))
        amt_wood = Feature('amt_wood', shape=[1], dtype='int', drange=slice(0, None))
        amt_gold = Feature('amt_gold', shape=[1], dtype='int', drange=slice(0, None))
        occupied_neighbors = Feature('occupied_neighbors', shape=[1], dtype='int', drange=slice(0, 9))
        num_townhall = sum(1 for a in self.agent_id_list if self.agent_class_map[a] == 1)

        townhall_obs_domain = ObservationDomain([amt_workers, amt_wood, amt_gold, occupied_neighbors],
                                                num_agents=num_townhall)

        # Worker: for the k nearest resources
        # (y, x) offsets, and their type (0: none, 1: wood, 2: gold) and their remaining counts (wood, gold)
        # & the worker's own remaining carry space (how many MORE resource it can carry)
        # & the player's amount of wood and gold
        # & occupancy of neighboring squares and actual content of those squares
        worker_items = []
        offset_range = max(self.rows, self.cols)
        for i in range(self.resources_visible):
            resource_offset = Feature(name=i, shape=[2], dtype='int', drange=slice(-offset_range, offset_range),
                                      prefix='resource_offset')
            resource_type = Feature(name=i, shape=[1], dtype='int', drange=slice(0, 3), prefix='resource_type')
            worker_items.append(resource_offset)
            worker_items.append(resource_type)

        carry_space = Feature(name='carry_space', shape=[2], dtype='int', drange=slice(0, self.worker_capacity + 1))
        worker_items.append(carry_space)

        player_wood = Feature(name='player_wood', shape=[1], dtype='int', drange=slice(0, None))
        worker_items.append(player_wood)

        player_gold = Feature(name='player_gold', shape=[1], dtype='int', drange=slice(0, None))
        worker_items.append(player_gold)

        # neighboring squares. start at top middle and go clockwise
        # -2: empty, -1: wall, 0: worker, 1: townhall, 2: resource
        neighbors = Feature(name='neighbors', shape=[8], dtype='int', drange=slice(-2, 3))
        worker_items.append(neighbors)

        num_workers = sum(1 for a in self.agent_id_list if self.agent_class_map[a] == 0)
        worker_obs_domain = ObservationDomain(worker_items, num_agents=num_workers)
        return {
            0: worker_obs_domain,
            1: townhall_obs_domain
        }

    def observe(self, obs_groups: List[Tuple[int, ...]] = None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        townhall_observation_domain = self.agent_class_observation_domains[1]
        worker_observation_domain = self.agent_class_observation_domains[0]

        resources = []

        for k in self.key_obj_map:
            if self.key_obj_map[k][0] == TYPE_RESOURCE:
                resources.append(k)

        observations = {}
        if obs_groups is None:
            #count num workers
            num_workers = 0
            for agent_id in self.agent_id_list:
                if self.agent_class_map[agent_id] == 0:
                    num_workers += 1
            for agent_id in self.agent_id_list:
                agent_class = self.agent_class_map[agent_id]
                if agent_class == 1 and self.allow_new_worker:
                    # townhall
                    observation = townhall_observation_domain.generate_empty_array()
                    observation[townhall_observation_domain.index_for_name('amt_workers')] = num_workers
                    observation[townhall_observation_domain.index_for_name('amt_wood')] = self.current_wood
                    observation[townhall_observation_domain.index_for_name('amt_gold')] = self.current_gold

                    obj_type, accept_wood, accept_gold, x, y = self.key_obj_map[agent_id]
                    neigbors_occupied = 0
                    for xi in range(x - 1, x + 2):
                        for yi in range(y - 1, y + 2):
                            if yi != y or xi != x:
                                if self.map[yi, xi] != 0:
                                    neigbors_occupied += 1

                    observation[townhall_observation_domain.index_for_name('occupied_neighbors')] = neigbors_occupied
                    observations[agent_id] = observation
                elif agent_class == 0:
                    #worker
                    observation = worker_observation_domain.generate_empty_array()
                    obj_type, amt_wood, amt_gold, x, y = self.key_obj_map[agent_id]
                    # find k nearest resources.
                    h = [] #keep the k nearest resources in decreasing order of distance
                    for r in resources:
                        _, res_wood, res_gold, xi, yi = self.key_obj_map[r]
                        dist = min(abs(x-xi, y-yi))
                        if len(h) < self.resources_visible:
                            heappush(h, (-dist, (r, res_wood, res_gold, xi, yi)))
                        else:
                            heappushpop(h, (-dist, (r, res_wood, res_gold, xi, yi)))

                    for i in range(self.resources_visible):
                        if i >= len(h):
                            break
                        r, res_wood, res_gold, xi, yi = h[i]
                        observation[worker_observation_domain.index_for_name(name=i, prefix='resource_offset')] = [yi, xi]

                        if res_wood+res_gold <= 0:
                            res_type = 0
                        elif res_wood >= res_gold:
                            res_type = 1
                        else:
                            res_type = 2

                        observation[worker_observation_domain.index_for_name(name=i, prefix='resource_type')] = res_type

                    observation[worker_observation_domain.index_for_name(name='carry_space')] = [self.worker_capacity['wood'] - amt_wood, self.worker_capacity['gold'] - amt_gold]
                    observation[worker_observation_domain.index_for_name(name='player_wood')] = self.current_wood
                    observation[worker_observation_domain.index_for_name(name='player_gold')] = self.current_gold


                    neighbor_types = []
                    for xi in range(x - 1, x + 2):
                        for yi in range(y - 1, y + 2):
                            if yi != y or xi != x:
                                if self.map[yi, xi] == 0:
                                    neighbor_types.append(-2)
                                elif self.map[yi,xi] == -1:
                                    neighbor_types.append(-1)
                                else:
                                    type = self.key_obj_map[self.map[yi,xi]][0]
                                    neighbor_types.append(type)
                    observation[worker_observation_domain.index_for_name(name='neighbors')] = neighbor_types
                    observations[agent_id] = observation

            return observations







