
import random
from typing import List, Dict, Union, Any, Tuple
import numpy as np
from common import Properties
from common.domain_transfer import DomainTransferMessage
from config import ConfigItemDesc, Config
from domain import ActionDomain, ObservationDomain, DiscreteActionDomain
from domain.actions import DiscreteAction
from domain.features import CoordinateFeature, DiscreteFeature, BinaryFeature, VectorFeature
from domain.hierarchical_domain import action_hierarchy_from_config, HierarchicalActionDomain
from environment.HierarchicalEnvironment import HierarchicalEnvironment

def _valid_wargus_map(data):
    assert isinstance(data, dict)
    world_size = data['world_size']
    assert isinstance(world_size, int)
    for f in data['forest_locations']:
        assert f[0] < world_size
        assert f[1] < world_size
    for g in data['goldmine_locations']:
        assert g[0] < world_size
        assert g[1] < world_size
    assert data['townhall_location'][0] < world_size
    assert data['townhall_location'][1] < world_size

    return True


def _valid_wargus_reward(data):
    assert isinstance(data, dict)
    for key in ['per_action', 'per_move', 'minegold', 'chopwood', 'deposit', 'error',
                'task_completion']:
        assert isinstance(data[key], int) or isinstance(data[key], float)
    return True


class Wargus(HierarchicalEnvironment):
    """
    The Wargus domain consists of collecting wood and gold resources.
    The goal is to collect a certain amount of each resource and deliver them to the townhall
    The gold mine and forest locations are fixed for every episode, though the starting location of the
    agent is randomly assigned every episode
        The locations are set with the 'goldmine_locations', 'forest_locations'

    The rewards can be set using the config, though the recommendation is -1 for all actions,
    -10 for illegal actions and 50 for completing a resource requirement

    The config also determines the amount of gold and wood to be collected,
        also determines how much wood and gold is held by each resource

        If the required amount of wood/gold exceeds the amount in a single forest/mine, the agent
            must learn to visit multiple resource locations

    Can be used with a task hierarchy to test HRL

    @author Robbie Dozier, grd27@case.edu (modified by Eric Miller, edm54@case.edu)
    """

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc(name="action_hierarchy",
                           check=lambda i: isinstance(i, str),
                           info='Location of action hierarchy config',
                           nestable=True),
            ConfigItemDesc(name='map',
                           check=_valid_wargus_map,
                           info='See _valid_wargus_map'),
            ConfigItemDesc(name='reward',
                           check=_valid_wargus_reward,
                           info='See _valid_wargus_reward'),
            ConfigItemDesc(name='required_gold',
                           check=lambda i: isinstance(i, int),
                           info='Gold goal for episode.'),
            ConfigItemDesc(name='required_wood',
                           check=lambda i: isinstance(i, int),
                           info='Wood goal for episode.'),
            ConfigItemDesc(name='gold_per_mine',
                           check=lambda i: isinstance(i, int),
                           info='Amount of gold in each mine'),
            ConfigItemDesc(name='wood_per_tree',
                           check=lambda i: isinstance(i, int),
                           info='Amount of wood in each forest')
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_addition=False,
                          use_agent_deletion=False)

    def __init__(self, config: Config):
        config = config.find_config_for_instance(self)

        # Boilerplate code for multi-agent once it gets implemented
        self.n = 1
        self._agent_id_list = list(range(self.n))
        self.agent_class = 0
        self._agent_class_map = {agent: self.agent_class for agent in self._agent_id_list}

        map_dict = config.map
        self.map_dict = map_dict
        self.world_size = map_dict['world_size']
        self.forest_locations = map_dict['forest_locations']
        self.goldmine_locations = map_dict['goldmine_locations']
        self.townhall_location = map_dict['townhall_location']

        self.required_gold = config.required_gold
        self.required_wood = config.required_wood
        self.gold_per_mine = config.gold_per_mine
        self.wood_per_tree = config.wood_per_tree

        self.reward_dict = config.reward

        # Check if the agent system is flat or hierarchical
        self._flat = True

        config.seed = 2
        np.random.seed()
        super(Wargus, self).__init__(config)

        hierarchy_config_dict = config.action_hierarchy
        if hierarchy_config_dict:
            action_hierarchy = action_hierarchy_from_config(hierarchy_config_dict)
            self._flat = False

        # Set up hierarchical domain
        if not self._flat:
            self.hierarchy = self.load_hierarchy(action_hierarchy)
            self.hierarchical_action_domain = HierarchicalActionDomain(self.hierarchy.root, self.hierarchy)
            self.hierarchical_observation_domain = self.abstract_all_observation_domains()

        # Set initial state
        # Resource: 0 - None; 1 - Gold; 2 - Wood
        self._state, self._meta_state = None, None
        self._reset_state()

    def load_hierarchy(self, action_hierarchy):
        state_var_values = {}
        actions = action_hierarchy.actions

        # Anything that can be passed as a parameter contributes to the number of grounded actions
        # To save memory, only enumerate values for state variables mentioned in params
        possible_params = set()

        for edge_variables in action_hierarchy.edges.values():
            for variable in edge_variables:
                possible_params.add(variable)

        bound_variables_map = {}

        if 'gold_loc' in possible_params:
            locations = self.map_dict['goldmine_locations']
            state_var_values['gold_loc'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'wood_loc' in possible_params:
            locations = self.map_dict['forest_locations']
            state_var_values['wood_loc'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'townhall_loc' in possible_params:
            locations = [self.map_dict['townhall_location']]
            state_var_values['townhall_loc'] = [f'_{l[0]}_{l[1]}_' for l in locations]
            for l in locations:
                bound_variables_map[f'_{l[0]}_{l[1]}_'] = l

        if 'other_loc' in possible_params:
            locations = [[0, 4], [4, 0]]
            state_var_values['other_loc'] = [f'_{l[0]}_{l[1]}_' for l in locations]
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
                        # print(sv, param, grounded_actions[param], state_var_values)
                        grounded_actions[param][sv] = state_var_values[sv]
                actions[action]['grounded_actions'] = grounded_actions

        return action_hierarchy.compile(state_var_values, bound_variables_map)

    def _reset_state(self, visualize: bool = False) -> Any:
        """
        Initialize the state with the config.
        :return: Initial state
        """

        # Assign starting location as a random space on the map
        start_location = [random.randrange(0, self.world_size), random.randrange(0, self.world_size)]

        self._state = {
            "location": start_location,
            "resource": 0,
            "gold_in_region": 0,
            "wood_in_region": 0,
            "townhall_in_region": 0,
            "meet_gold_requirement": 0,
            "meet_wood_requirement": 0,
            "wood_remaining":0,
            "gold_remaining":0
        }

        self._state["gold_in_region"] = int(self._state["location"] in self.goldmine_locations)
        self._state["wood_in_region"] = int(self._state["location"] in self.forest_locations)
        self._state["wood_in_region"] = int(self._state["location"] == self.townhall_location)

        self._meta_state = {
            "gold": 0,
            "wood": 0,
            "goldmine_state": {tuple(k): self.gold_per_mine for k in self.goldmine_locations},
            "forest_state": {tuple(k): self.wood_per_tree for k in self.forest_locations}
        }

        state = np.zeros(5 + len(self.forest_locations) + len(self.goldmine_locations))

        state[0] = self._state["location"][0]
        state[1] = self._state["location"][1]
        state[2] = self._state["resource"]
        state[3] = self._state["meet_gold_requirement"]
        state[4] = self._state["meet_wood_requirement"]

        # Add variables for wood and gold remaining in forest and mines
        state[5:5+len(self.forest_locations)] = self.wood_per_tree
        state[-1 * len(self.goldmine_locations):] = self.gold_per_mine

        return state

    def _create_action_domains(self, config) -> Dict[Union[str, int], ActionDomain]:
        return {self.agent_class: DiscreteActionDomain([DiscreteAction("wargus_actions", 7)], 1)}

    def _create_observation_domains(self, config) -> Dict[Union[str, int], ObservationDomain]:
        location = CoordinateFeature(name='location', lower=[0, 0],
                                     upper=[self.world_size, self.world_size],
                                     is_discrete=True)
        resource = DiscreteFeature(name="resource", size=3)
        meet_gold_requirement = BinaryFeature(name="meet_gold_requirement")
        meet_wood_requirement = BinaryFeature(name="meet_wood_requirement")

        items = [location, resource, meet_gold_requirement, meet_wood_requirement]

        # Add variables that track the gold and wood left in each mine/forrest
        for forest in range(len(self.forest_locations)):
            wood_remaining = BinaryFeature(name=f'{forest}', prefix='wood_remaining')
            items.extend([wood_remaining])

        for mine in range(len(self.goldmine_locations)):
            wood_remaining = BinaryFeature(name=f'{mine}', prefix='gold_remaining')
            items.extend([wood_remaining])

        self._observation_domain = ObservationDomain(items, num_agents=1)
        return {self.agent_class: self._observation_domain}

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('wargus_actions')
        for agent in actions:
            reward = 0
            action = actions[agent][action_index][0]
            if action in {0, 1, 2, 3}:

                # Move action
                move_map = [
                    [0, 1],   # North
                    [0, -1],  # South
                    [1, 0],   # East
                    [-1, 0]   # West
                ]

                # apply randomness with 20% chance of different move action
                actual_action = action
                if random.random() < 0.2:
                    # Move agent in a perpendicular direction
                    if random.random() < 0.5:
                        # Move left
                        actual_action = [3, 2, 0, 1][action]
                    else:
                        # Move right
                        actual_action = [2, 3, 1, 0][action]

                current_position = self._state['location']
                target = [
                    current_position[0] + move_map[actual_action][0],
                    current_position[1] + move_map[actual_action][1]
                ]
                # Check if in bounds
                if target[0] < 0 or target[0] >= self.world_size or target[1] < 0 or target[1] >= self.world_size:
                    # Out of bounds, don't move
                    target = current_position

                # Execute move
                self._state['location'] = target
                reward = self.reward_dict['per_move']

            elif action in [4, 8]:
                # Mine gold
                # Check to see if goldmine in range
                in_range = self._state['location'] in self.goldmine_locations

                # If in range and not currently carrying anything
                if in_range and self._state['resource'] == 0:

                    # If there is gold left in the mine
                    if self._meta_state['goldmine_state'][tuple(self._state['location'])] > 0:
                        # Mine the gold
                        self._state['resource'] = 1
                        self._meta_state['goldmine_state'][tuple(self._state['location'])] -= 1
                        reward = self.reward_dict['minegold']
                    else:
                        # Illegal mine action --> no gold left to mine
                        reward += self.reward_dict['error']

                else:
                    # Illegal mine action --> not at a mine or carrying something already
                    reward += self.reward_dict['error']

            elif action in [5, 9, 7]:
                # Chop wood
                # Check to see if forest in range
                in_range = self._state['location'] in self.forest_locations

                # If in range and not currently carrying anything
                if in_range and self._state['resource'] == 0:
                    # If there is wood left in the forest
                    if self._meta_state['forest_state'][tuple(self._state['location'])] > 0:
                        # Chop the wood
                        self._state['resource'] = 2
                        self._meta_state['forest_state'][tuple(self._state['location'])] -= 1
                        reward = self.reward_dict['chopwood']

                    else:
                        # Illegal action --> no wood left to chop
                        reward += self.reward_dict['error']

                else:
                    # Illegal action --> not at a forest or carrying something already
                    reward += self.reward_dict['error']

            elif action == 6 or action == 10:
                # Deposit
                # Check to see if town hall in range
                in_range = self._state['location'] == self.townhall_location
                reward += self.reward_dict['deposit']

                # If in range and currently carrying something
                if in_range and self._state['resource'] != 0:
                    # Make deposit
                    if self._state['resource'] == 1:
                        # Gold
                        self._meta_state["gold"] += 1
                        self._state['resource'] = 0

                        # If requirement has already been fulfilled, assign error reward
                        if self._state['meet_gold_requirement']:
                            reward += self.reward_dict['error']

                    elif self._state['resource'] == 2:
                        # Wood
                        self._meta_state["wood"] += 1
                        self._state['resource'] = 0

                        # If requirement has already been fulfilled, assign error reward
                        if self._state['meet_wood_requirement']:
                            reward += self.reward_dict['error']
                else:
                    # Illegal action --> depositing out of range or nothing to deposit
                    reward += self.reward_dict['error']

            # Check to see if requirements have been fulfilled
            if self._meta_state["gold"] >= self.required_gold and not self._state['meet_gold_requirement']:
                reward += self.reward_dict['task_completion']
                self._state['meet_gold_requirement'] = 1

            if self._meta_state["wood"] >= self.required_wood and not self._state['meet_wood_requirement']:
                reward += self.reward_dict['task_completion']
                self._state['meet_wood_requirement'] = 1

            # Apply reward
            rewards[agent] = reward

            # Terminate the episode
            if self._state['meet_gold_requirement'] and self._state['meet_wood_requirement']:
                self.done = True

            # Update gold/wood in region
            self._state["gold_in_region"] = int(list(self._state["location"]) in [
                list(k) for k, v in self._meta_state["goldmine_state"].items() if v > 0])
            self._state["wood_in_region"] = int(list(self._state["location"]) in [
                list(k) for k, v in self._meta_state["forest_state"].items() if v > 0])
            self._state["townhall_in_region"] = int(list(self._state["location"]) == self.townhall_location)
                
        return rewards

    def observe(self, obs_groups: List[Tuple[int, ...]] = None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        domain = self._observation_domain
        observation = domain.generate_empty_array()
        observation[domain.index_for_name("location")] = self._state['location']
        observation[domain.index_for_name("resource")] = self._state['resource']
        observation[domain.index_for_name("meet_gold_requirement")] = self._state['meet_gold_requirement']
        observation[domain.index_for_name("meet_wood_requirement")] = self._state['meet_wood_requirement']

        # Add wood and gold remaining state variables
        for ind, forest in enumerate(self.forest_locations):
            observation[domain.index_for_name("wood_remaining_" + str(ind))] = self._meta_state['forest_state'][tuple(forest)] > 0

        for ind, mine in enumerate(self.goldmine_locations):
            observation[domain.index_for_name("gold_remaining_" + str(ind))] = self._meta_state['goldmine_state'][tuple(mine)] > 0

        return {0: observation}

    def abstracted_observation_domain(self, state_variables: set) -> ObservationDomain:
        """
        Observation domain contains the state variables for Wargus
        Holds only the state variables that are required at this node in the hierarchy
        :return: Observation Domain
        """
        items = []

        if 'location' in state_variables:
            # The agent's position
            location = CoordinateFeature(name='location', lower=[0, 0],
                                         upper=[self.world_size, self.world_size],
                                         is_discrete=True)
            items.append(location)

        if 'resource' in state_variables:
            resource = DiscreteFeature(name="resource", size=3)
            items.append(resource)

        if 'meet_gold_requirement' in state_variables:
            meet_gold_requirement = BinaryFeature(name="meet_gold_requirement")
            items.append(meet_gold_requirement)

        if 'meet_wood_requirement' in state_variables:
            meet_wood_requirement = BinaryFeature(name="meet_wood_requirement")
            items.append(meet_wood_requirement)

        if "wood_remaining" in state_variables:
            for forest in range(len(self.forest_locations)):
                wood_remaining = BinaryFeature(name=f'{forest}', prefix='wood_remaining')
                items.extend([wood_remaining])

        if "gold_remaining" in state_variables:
            for mine in range(len(self.goldmine_locations)):
                gold_remaining = BinaryFeature(name=f'{mine}', prefix='gold_remaining')
                items.extend([gold_remaining])

        return ObservationDomain(items, num_agents=self.n)

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[int]:
        return [0]

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    def transfer_domain(self, message: DomainTransferMessage) -> DomainTransferMessage:
        pass

    # NO RNG
    def set_initial_seed(self, seed: int):
        pass

    def set_seed_state(self, seed_state):
        pass

    def get_seed_state(self):
        pass
