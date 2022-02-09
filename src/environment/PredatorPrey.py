"""
A Predator-Prey environment.

Includes specific formulations for multi-agent learning.

Includes joint features and action-tied features.

Rewards are shared among all agents.

Though OpenAI Multiagent PredatorPrey was considered, it is inefficient to expect users to install a separate project
from github or use another project which requires script calls outside of a specific package model.

For that reason, this is implemented separately.
"""
from collections import OrderedDict
from typing import Tuple, List, Dict, Union, TYPE_CHECKING

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common.aux_env_info import AuxiliaryEnvInfo
from common.domain_transfer import DomainTransferMessage
from config import checks
from config.config import ConfigItemDesc
from domain.actions import DiscreteAction
from environment import HandcraftEnvironment
from common.properties import Properties
from domain.DiscreteActionFeatureDomain import DiscreteActionFeatureDomain
from domain.observation import ObservationDomain
from domain.DiscreteActionDomain import DiscreteActionDomain
from domain.ActionDomain import ActionDomain
from domain.features import Feature
from common.vecmath import vec_choice, CachedOps

if TYPE_CHECKING:
    from config import Config

# Value on map
PREDATOR_VALUE = 1
PREY_VALUE = 2


def _random_or_positions_in_map(l):
    ret = isinstance(l, list) and len(l) > 0
    for dim in l:
        assert dim == 'random' or all([isinstance(coord, int) and coord >= 0 for coord in dim])
    return ret


class PredatorPrey(HandcraftEnvironment):

    @classmethod
    def __valid_prey_ai(cls, method, self=None):
        src = cls if self is None else self
        return {
            'frozen': src.__prey_dirs_frozen,
            'random': src.__prey_dirs_random,
            'greedy_escape': src.__prey_dirs_greedy_escape,
            'softmax_escape': src.__prey_dirs_softmax_escape
        }[method]

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc('map_shape',
                           checks.positive_int_list,
                           info='A list of positive ints describing the shape of the map rectangle.'),
            ConfigItemDesc('predators',
                           _random_or_positions_in_map,
                           info='A list of int coordinates or "random" '
                                'describing predator positions in Agent ID order'),
            ConfigItemDesc('prey',
                           _random_or_positions_in_map,
                           info='A list of int coordinates or "random" '
                                'describing prey positions.'),
            ConfigItemDesc('prey_ai',
                           lambda s: callable(cls.__valid_prey_ai(s)),
                           info='Describes simple prey AI. May be "frozen" or "random" or "greedy_escape" for '
                                'respective non-moving, random movement or greedy escape from the nearest predator.'),
            ConfigItemDesc('share_space',
                           lambda b: isinstance(b, bool),
                           info='Whether or not predators are allowed to share space in the map.'),
            ConfigItemDesc('capture_requirement',
                           checks.positive_integer,
                           info='The number of agents which must surround a prey for the prey to be captured. '
                                'Typically this is the dimensionality of the map.'),
            ConfigItemDesc('allow_noop',
                           lambda b: isinstance(b, bool),
                           info='Whether or not to allow predators to skip turns.'),
            ConfigItemDesc('capture_reward', checks.numeric, info='Reward for capturing prey.'),
            ConfigItemDesc('chase_reward', checks.numeric, info='Reward for approaching prey. If nonzero, calculates '
                                                                'a shaped reward for the agents. Otherwise, the '
                                                                'environment produces a sparse reward.'),
            ConfigItemDesc('time_reward', checks.numeric, info='Reward (usually negative) for taking a turn..'),
            ConfigItemDesc('collision_reward', checks.numeric, info='Reward (usually negative) for agents colliding '
                                                                  'with each other if share_space is false, or walls.'),
            ConfigItemDesc('prey_temp', lambda x: checks.positive_float(x) or x is 'mixture',
                           info='Fixed softmax temperature for prey softmax escape rule-based algorithm. Can be '
                                'a positive float or "mixture" for random.',
                           optional=True, default='mixture')
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=True,
                          use_agent_deletion=True,   # TODO wip
                          use_agent_addition=False)  # TODO wip

    @property
    def agent_class_list(self) -> List[str]:
        return ['predator']

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    def __init__(self, config: 'Config'):
        """
        Initialize the Predator Prey environment.

        Config expects the following:

        :param config: Configuration
        """
        env_cfg = config.find_config_for_instance(self)
        self.share_space = env_cfg.share_space
        self.capture_requirement = env_cfg.capture_requirement
        self.allow_noop = env_cfg.allow_noop
        self.prey_ai = env_cfg.prey_ai
        self.prey_temp = env_cfg.prey_temp

        self.chase_reward = env_cfg.chase_reward
        self.capture_reward = env_cfg.capture_reward
        self.time_reward = env_cfg.time_reward
        self.collision_reward = env_cfg.collision_reward

        self.map_shape: np.ndarray = env_cfg.map_shape
        self.dims = len(self.map_shape)

        # Construct directions along the axes
        dirs = np.zeros((self.dims * 2, self.dims), dtype='int8')
        for i in range(dirs.shape[0]):
            dirs[i, i // 2] = ((i % 2) * 2) - 1
        # Append the zero-direction if no-op allowed
        if self.allow_noop:
            dirs = np.vstack((np.zeros(self.dims, dtype='int8'), dirs))
        self.directions = dirs

        # Joint directions: [dir, 2, dims]
        self.joint_directions = np.stack(DiscreteActionFeatureDomain.broadcast_joint(dirs, dirs), axis=1)
        self.num_jdirs = self.joint_directions.shape[0]
        # n-joint directions (joint for n agents): [dir, n, dims]
        n_directions = self.joint_directions
        for _ in range(2, self.capture_requirement):
            n_directions = np.stack(DiscreteActionFeatureDomain.broadcast_joint(dirs, n_directions), axis=1)
        self.n_directions = n_directions

        self._move_prey = PredatorPrey.__valid_prey_ai(env_cfg.prey_ai, self)

        # Used to initialize state
        self.config_predators: List[List] = env_cfg.predators
        self.config_prey: List[List] = env_cfg.prey

        # Allocate variables
        self.map = np.array([])
        self.predators = np.array([])
        self.predators_next = np.array([])
        self._agent_id_list = []
        
        super().__init__(config)
        self.joint_agent_class_observation_domains = self._create_joint_agent_class_observation_domains(config)

    # This env uses numpy for randomness, the numpy seed is set in the controller
    def set_initial_seed(self, seed: int):
        pass

    def get_seed_state(self):
        return []

    def set_seed_state(self, seed_state):
        pass

    def get_auxiliary_info(self) -> AuxiliaryEnvInfo:
        return AuxiliaryEnvInfo(joint_observation_domains=self.joint_agent_class_observation_domains)

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        """
        Initialize / reset the environment state.
        """
        # The map just contains "what is here?" It is an arbitrary size.
        self.map = np.zeros(self.map_shape)
        free_squares = np.indices(self.map_shape).reshape(self.dims, self.map.size).T
        np.random.shuffle(free_squares)
        free_sample_idx = 0

        ### CREATE AGENTS
        # Prevent spawning pred or prey on top of each other.
        predators = []
        for predator in self.config_predators:
            if not isinstance(predator, list):
                predator = free_squares[free_sample_idx]
                free_sample_idx += 1
            predators.append(predator)
            self._put_in_map(predator, 'Predator', PREDATOR_VALUE)
        self.predators = np.array(predators)
        self.predators_next = np.copy(self.predators)       # Temp array for calculation

        ### Spawn prey from inverse euclidean distance to closest pred
        # [#, dims]
        left_squares = free_squares[free_sample_idx:]
        # [#] <- [#, preds] <- [#, preds, dims]
        min_pred_dists = np.square(left_squares[:, None] - self.predators).sum(axis=2).min(axis=1)
        chosen_squares = left_squares[np.random.choice(np.r_[:left_squares.shape[0]],
                                                       p=min_pred_dists/min_pred_dists.sum(),
                                                       size=len(self.config_prey),
                                                       replace=False)]

        all_prey = []
        free_sample_idx = 0
        for prey in self.config_prey:
            if not isinstance(prey, list):
                prey = chosen_squares[free_sample_idx]
                free_sample_idx += 1
            all_prey.append(prey)
            self._put_in_map(prey, 'Prey', PREY_VALUE)
        self.prey = np.array(all_prey)
        self.prey_alive = len(self.prey)
        # Note: Negative position indicates "dead" prey. Don't actually remove the prey. We put it in purgatory.
        self.prey_range = np.r_[:len(self.prey)]  # helper data

        self._agent_id_list = [i for i in range(0, len(self.predators))]
        self._agent_class_map = {agent_id: 'predator' for agent_id in self._agent_id_list}
        self.step = 0

        # Some hidden variables for "prey character" if we're using stochastic prey. Makes it more interesting.
        if self.prey_temp is 'mixture':
            self.prey_characteristic = np.random.rand(2, self.prey_alive)
            self.prey_characteristic[1] = np.minimum(self.prey_characteristic[1], 0.05) * 3
        else:
            self.prey_characteristic = np.ones((2, self.prey_alive))
            self.prey_characteristic[1] = self.prey_temp

        return self.map

    def _coord_hash(self, coord: np.ndarray) -> np.ndarray:
        """
        Take the hash of a coordinate.
        As long as the coordinate is within the map, the hash will be unique.
        :param coord: coordinate np.ndarray
        :return: The unique hash of the coord
        """
        return np.dot(coord, self.map.strides)

    def _put_in_map(self, coord: np.ndarray, name: str, value: int):
        try:
            self.map[tuple(coord)] = value
        except IndexError as e:
            print(e)
            raise ValueError('{} at position [{}] was out of bounds in map of shape [{}]. Please check '
                             'the environment configuration before running the experiment.'
                             .format(name, coord, self.map_shape))

    def _create_observation_domains(self, config) -> Dict[str, ObservationDomain]:

        # Will agent hit the boundary? (We are NOT going to make "PredatorPreyFunctionApproximator" - ...
        # The reference generates some features from joint (observation + action) - that is for the function approx.
        # Therefore, this offers one step down and produces similar features which don't depend directly on action.

        rbin = slice(0, 2)     # Binary range
        pms = slice(-1, 2)     # Plus-minus
        # Agent on edge. 1 if moving in that direction would bump into the edge. 0 otherwise.
        on_edge = Feature(name='1p_edge', shape=[1], dtype='int', drange=rbin)

        # For each prey, make 3 features; when we observe, these will actually be sorted in descending closeness
        # To the predator, prey are only differentiated by their relative distance.
        # So, the farther prey come first in the observation. The closer prey come up later.
        # The reason is that this saves some computation.
        features = [on_edge]
        for prey_count, prey in enumerate(self.prey):
            ord_prefix = '1p_prey_farthest_{}'.format(prey_count)
            # If moving the predator would bring it adjacent to the prey. Distinct from MD since it's special position.
            features.append(Feature(prefix=ord_prefix, name='adjacent', shape=[1], dtype='int', drange=rbin))
            # Modified distance to prey: (range [0, 1] inclusive)
            features.append(Feature(prefix=ord_prefix, name='distance', shape=[1], dtype='float', drange=rbin))
            # Whether the move causes predators to stay still relative to the prey, or get closer
            features.append(Feature(prefix=ord_prefix, name='still', shape=[1], dtype='int', drange=pms))

        obs_domain = ObservationDomain(features)
        act_domain: DiscreteActionDomain = self.agent_class_action_domains['predator']
        feature_domain = DiscreteActionFeatureDomain(obs_domain, act_domain)
        self.single_agent_packed_shape = feature_domain.packed_shape
        self.single_agent_ravel_shape = feature_domain.shape[0]

        return {
            'predator': feature_domain
        }

    def _create_joint_agent_class_observation_domains(self, config):
        # Just use ActionFreeLinear with this and you'll be fine.

        # Uses DiscreteActionFeatureDomain.

        predator_action = self.agent_class_action_domains['predator']
        joint_action = DiscreteActionDomain(predator_action.items * 2, num_agents=2)
        s_bin = slice(0, 2)

        # binary for "would joint action put me next to a non-pair agent for agent 0"
        nonpair_1 = Feature(name='2p_nonpair_0', shape=[1], dtype='int', drange=s_bin)

        # binary for "would joint action put me next to a non-pair agent for agent 1"
        nonpair_2 = Feature(name='2p_nonpair_1', shape=[1], dtype='int', drange=s_bin)

        # binary for "would joint action collide the pair of agents"
        pair_collide = Feature(name='2p_collide', shape=[1], dtype='int', drange=s_bin)

        # binary for "would joint action align the pair of agents
        pair_align = Feature(name='2p_align', shape=[1], dtype='int', drange=s_bin)

        # Predator pair prey features (2 continuous for each prey). Will be 0 if the prey is removed.
        # log_max = ln(2*max_corner_manhattan + 2); w = 1 - [ ln(d0 + d1 + 2) - ln(2) ] / log_max
        # both are closing, f=[w, w]; one stays same, f=[-w, w]; else [-w, -w]
        prey_feats = []
        for prey_count, prey in enumerate(self.prey):
            prey_feats.append(Feature(name='2p_chase_{}'.format(prey_count), shape=[2], dtype='float'))
            prey_feats.append(Feature(name='2p_adjacent_{}'.format(prey_count), shape=[2], dtype='int'))

        obs_domain = ObservationDomain([nonpair_1, nonpair_2, pair_collide, pair_align] + prey_feats)
        joint_domain = DiscreteActionFeatureDomain(obs_domain, joint_action)

        self.double_agent_packed_shape = joint_domain.packed_shape
        self.double_agent_ravel_shape = joint_domain.shape[0]

        return {('predator', 'predator'): joint_domain}

    def _create_action_domains(self, config) -> Dict[str, ActionDomain]:
        act_move = DiscreteAction(name='move', num_actions=len(self.directions))
        return {
            'predator': DiscreteActionDomain([act_move], len(self.predators))
        }

    # @profile
    def observe(self, obs_groups: List[Union[int, Tuple[int, ...]]]=None)\
            -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        # Not using domain slicers. Hardcoding this for speed. (probably)
        if obs_groups is None:
            single_obs = self.agent_id_list
            joint_obs = []
        else:
            single_obs = [agent_id for agent_id in obs_groups if isinstance(agent_id, int)]
            joint_obs = [agent_id for agent_id in obs_groups if isinstance(agent_id, tuple) and len(agent_id) > 1]

        curr_preds = self.predators[:, None, :]                 # [pred, ____, dims]
        next_preds = curr_preds + self.directions[None, ...]    # [pred, dirs, dims]
        prey_capt = (self.prey < 0).any(axis=1)                 # [prey]

        ## Predator-prey distances are calculated for all pairs.
        # [pred, dirs, prey] : manhattan_dist <- (pred, dirs, ____, dims) & (____, dirs, prey, dims)
        curr_pdist = np.sqrt(np.square(curr_preds[:, :, None, :] - self.prey[None, ...]).sum(axis=3))
        next_pdist = np.sqrt(np.square(next_preds[:, :, None, :] - self.prey[None, ...]).sum(axis=3))

        # It's unfair to disregard the predator-prey collision which can cause predators to get backed up.
        pred_prey_collide = (next_pdist == 0).any(axis=2)       # [pred, dirs]
        for pred, collides in enumerate(pred_prey_collide.any(axis=1)):
            if collides.any():
                next_preds[pred, collides] = curr_preds[pred, 0]
        next_pdist[next_pdist == 0] = 1

        # [prey] : maximum manhattan distance from edge
        edge_dist = np.maximum(self.prey, self.map_shape - self.prey - 1).sum(axis=1)

        o = self.__observe_single(single_obs, next_preds, curr_pdist, next_pdist, edge_dist, prey_capt)
        ret = {agent_id: o[i].flatten() for i, agent_id in enumerate(single_obs)}

        if joint_obs:
            log_edge_dist = np.log(edge_dist * 2 + 2)       # calc log_max outside here
            pairs = np.array(joint_obs)
            joint_o = self._observe_pairs(pairs, next_preds, curr_pdist, next_pdist, log_edge_dist, prey_capt)
            for i, pair in enumerate(joint_obs):
                ret[pair] = joint_o[i]
        return ret

    # @profile
    def __observe_single(self, single_obs, next_preds, curr_pdist, next_pdist, edge_dist, prey_capt):
        # o for observation: [AGENT, ACTION, FEATURE]
        o = np.zeros((len(single_obs),) + self.single_agent_packed_shape, dtype='float32')
        next_preds_single = next_preds[single_obs]
        sel_next_pdist = next_pdist[single_obs]
        sel_curr_pdist = curr_pdist[single_obs]
        # on-edge: move takes us out out the map. -> [pred, dirs]
        o[:, :, 0] = (next_preds_single < 0).any(axis=2) | (next_preds_single >= self.map_shape).any(axis=2)
        prey_feats = o[:, :, 1:].reshape(next_pdist.shape + (3,))    # [pred, dirs, prey, feat]
        # Adjacent: 1 if predator would move next to prey.
        prey_feats[..., 0] = next_pdist == 1
        # Modified distance: Increases mostly with manhat dist but also "centrality" of prey.
        prey_feats[..., 1] = np.square(1 - np.log(sel_next_pdist*0.5 + 1) / np.log(edge_dist[None, None, :] + 2))
        # Still feature: Pred-prey dist remains same.
        prey_feats[..., 2] = sel_curr_pdist == sel_next_pdist
        if prey_capt.any():  # Zero out features for captured prey
            prey_feats[:, :, prey_capt] = 0
        # then sort along modified distances so closer prey are last (we order ascending closeness to save the minus)
        prey_feats[:] = prey_feats[np.arange(prey_feats.shape[0])[:, None, None],
                                   np.arange(prey_feats.shape[1])[None, :, None],
                                   np.argsort(prey_feats[..., 1], axis=2)]
        return o

    # @profile
    def _observe_pairs(self, pairs, next_preds, curr_pdist, next_pdist, log_edge_dist, prey_capt):
        joint_o = np.zeros((len(pairs),) + self.double_agent_packed_shape, dtype='float32')
        joint_next_preds = next_preds[np.array(pairs)]
        # pairnum, dirs, dims
        njpreds = CachedOps.broadcast_joint_3d_stack2(joint_next_preds[:, 0], joint_next_preds[:, 1])
        pair_equals = np.equal(njpreds[:, :, 0], njpreds[:, :, 1])
        # nonpair (2): Predator moves close to non-coordinating agent
        njpreds_close = np.abs(njpreds[..., None, :] - self.predators[None, None, None]).sum(axis=4) <= 1
        njpreds_close[np.arange(len(pairs))[:, None], :, :, pairs] = False
        joint_o[..., 0:2] = njpreds_close.any(axis=3)
        # Pair collision (1): Pair will collide
        joint_o[..., 2] = pair_equals.all(axis=2)
        # Pair alignment (1): Pair aligns on any axis
        joint_o[..., 3] = pair_equals.any(axis=2)
        # [pairs, jdirs, prey, size_preyfeats]
        jp_feats = joint_o[..., 4:].reshape(len(pairs), self.num_jdirs, self.prey.shape[0], 4)
        next_jpdist = next_pdist[pairs]
        next_pdist2 = CachedOps.broadcast_joint_3d_stack2(next_jpdist[:, 0], next_jpdist[:, 1])
        curr_pdist2 = curr_pdist[pairs, 0][:, None]

        # pairchase (2 * prey) ;; Might want to sort these by prey if it still doesn't turn well...?
        modified_distance = 1 - (np.log(next_pdist2.sum(axis=2) + 2) - np.log(2)) / log_edge_dist[None, None]
        np.square(modified_distance, out=modified_distance)
        jp_feats[..., 0:2] = modified_distance[..., None]
        jp_feats[..., 2] = (next_pdist2 == 1).all(axis=2)
        jp_feats[..., 3] = (next_pdist2 <= 2).all(axis=2)

        accel = np.sign(curr_pdist2 - next_pdist2).sum(axis=2)
        np.negative(jp_feats[..., 0], where=accel < 2, out=jp_feats[..., 0])
        np.negative(jp_feats[..., 1], where=accel < 1, out=jp_feats[..., 1])
        jp_feats[:, :, prey_capt] = 0
        jp_feats[:] = jp_feats[np.arange(jp_feats.shape[0])[:, None, None],
                               np.arange(jp_feats.shape[1])[None, :, None],
                               np.argsort(modified_distance, axis=2)]

        return joint_o

#     # @profile
    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        # PredatorPrey has a single, global reward.
        reward = self.time_reward

        # All agents move simultaneously to reduce strange, biased collision resolution.
        # This way, if a group of agents move onto the same square in the same turn when space cannot be shared,
        # All the agents are prevented from making that move.
        # If the square is competitive, a random agent of the group is chosen.
        reward += self._move_predators(actions)
        reward += self._capture_prey()

        if self.chase_reward != 0:
            reward += self._calculate_chase_reward()

        if self.prey_alive <= 0:
            self.done = True
        else:
            self._move_prey()

        return {agent_id: reward for agent_id in self.agent_id_list}

#     # @profile
    def _move_predators(self, actions) -> int:
        # TODO (future) Accelerate in future, maybe? It's definitely not a bottleneck for now.
        accumulated_reward = 0
        tile_hashes = {}
        # Agents that collide into each other or the wall and therefore should not move
        revert_agents = np.zeros(len(self.predators), dtype=bool)
        for agent_id, action in actions.items():
            # action is the index of the direction the agent moves in
            direction = action[0]
            move_to = self.predators[agent_id] + self.directions[direction]
            coord_hash = self._coord_hash(move_to)
            if np.greater_equal(move_to, self.map_shape).any() or (move_to < 0).any():
                # Bumped into a wall or prey and shouldn't move there
                revert_agents[agent_id] = True
                accumulated_reward += self.collision_reward
            elif self.map[tuple(move_to)] != 0:     # hit_prey. prey can't share pred anyway
                # repeated in an elif since the check throws indexerror if out of bounds
                revert_agents[agent_id] = True
                accumulated_reward += self.collision_reward
            elif not self.share_space and coord_hash in tile_hashes:
                # The agent which we bumped into on the square cannot actually move there anymore.
                # TODO (future) stochastic under competitive square
                revert_agents[agent_id] = True
                collided_agent_id = tile_hashes[coord_hash]
                revert_agents[collided_agent_id] = True
                accumulated_reward += self.collision_reward
            else:
                # Move was ok or we are first there before collision (and will have to revert)
                tile_hashes[coord_hash] = agent_id
                self.predators_next[agent_id] = move_to

        for agent_id, revert in enumerate(revert_agents):
            if not revert:
                self.map[tuple(self.predators[agent_id])] = 0
                self.map[tuple(self.predators_next[agent_id])] = PREDATOR_VALUE

        # Revert agents which couldn't actually move
        self.predators_next[revert_agents] = self.predators[revert_agents]
        # Swap! Since dummy_predators is now the newest, recycling the array. This saves computation.
        temp = self.predators
        self.predators: np.ndarray = self.predators_next
        self.predators_next: np.ndarray = temp

        return accumulated_reward

    def _capture_prey(self) -> int:
        # Capture prey first
        # TODO (future...) could prevent calculation on prey in purgatory
        # TODO (benchmark) check: using map or numpy faster?
        surround_count = (abs(self.prey[:, None, :] - self.predators[None, :, :]).sum(axis=2) == 1).sum(axis=1)
        capture_mask = np.greater_equal(surround_count, self.capture_requirement)
        capture_num = capture_mask.sum()
        # Send captured prey to purgatory. Luckily, prey in purgatory cannot have neighbors.
        if capture_num > 0:
            for prey in self.prey[capture_mask]:
                self.map[tuple(prey)] = 0
        self.prey[capture_mask] = -1
        self.prey_alive -= capture_num
        # if capture_num > 0:
        #     print('Captured {} prey for {} left!'.format(capture_num, self.prey_alive))

        return self.capture_reward * capture_num

    def _move_prey(self) -> int:
        raise NotImplementedError('Function should be overwritten during init time via prey_ai config.')

    def __prey_dirs_frozen(self):
        # Literally don't move.
        pass

    def __prey_dirs_random(self):
        # Move prey randomly.
        random_directions = self.directions[np.random.randint(len(self.directions), size=self.prey_alive)]
        self.__apply_prey_move(random_directions)

    def __prey_dirs_greedy_escape(self):
        """
        Move the prey away from the closest agent.
        If multiple agents are equally close, the prey will randomly pick one to move away from.
        """
        # Offsets as [Prey, Pred, dim]
        offsets = self.prey[:, None, :] - self.predators[None, :, :]
        # Manhattan distances for pred-prey pairs as [Prey, Pred]
        manhattan_distances = abs(offsets).sum(axis=2)
        # Closest predator for each. Picks the first predator if two predators are equally close.
        closest = np.argmin(manhattan_distances, axis=1)
        closest_offsets = offsets[self.prey_range, closest]
        direction_similarity = np.maximum(0, np.dot(closest_offsets, self.directions.T))
        preferred_directions = vec_choice(direction_similarity, self.directions)
        self.__apply_prey_move(preferred_directions)

    def __prey_dirs_softmax_escape(self):
        """
        Move the prey based on a mixture over distributions of predator distances
        """
        alive_mask = (self.prey >= 0).all(axis=1)
        living_prey = self.prey[alive_mask, :]
        preyc = self.prey_characteristic[:, alive_mask]
        directions = np.zeros((self.prey.shape[0], self.dims), dtype=np.uint8)
        next_prey = living_prey[:, None] + self.directions[None]
        eucl_dist = np.sqrt((np.square(next_prey[:, :, None] - self.predators[None, None]) /
                             max(self.map_shape)).sum(axis=3))
        eucl_dist.sort(axis=2)
        f_closest = eucl_dist[..., 0]
        f_squeeze = np.abs(eucl_dist[..., 0] - eucl_dist[..., 1])
        pref1 = preyc[0, :, None]
        pref2 = preyc[1, :, None]
        prob = pref1 * f_closest + (1 - pref1) * f_squeeze
        prob /= pref2
        prob -= prob.max(axis=1, keepdims=True)
        preferred_directions = vec_choice(np.exp(prob), self.directions)
        directions[alive_mask] = preferred_directions
        self.__apply_prey_move(preferred_directions)

#     # @profile
    def __apply_prey_move(self, random_directions):
        i = 0
        for prey in self.prey:
            # If prey alive
            if not (prey < 0).any():
                next_pos = prey + random_directions[i]
                i += 1
                if np.less(next_pos, self.map_shape).all() \
                        and (next_pos > 0).all() \
                        and self.map[next_pos[0], next_pos[1]] == 0:
                    # First prey there gets the square.
                    # Prey cannot use "share_space"
                    self.map[prey[0], prey[1]] = 0
                    prey[:] = next_pos
                    self.map[next_pos[0], next_pos[1]] = PREY_VALUE

    def _calculate_chase_reward(self):
        # TODO (future) - Not a typical reward in these environments. Makes it possibly greedy.
        raise NotImplementedError()

    def visualize(self):
        # TODO (future) Switch to pyglet visualization.
        # Also, proooooobably should make sure map isn't something other than 2D.
        visual = np.copy(self.map)
        cmap = ListedColormap(colors=['w', 'r', 'b'])
        if not hasattr(self, 'fig'):
            self.fig = plt.figure('env', figsize=(3, 3))
            plt.ion()
        if not hasattr(self, '_image'):
            fig = plt.figure('env', figsize=(3, 3))
            self._fig = fig
            self._image = plt.imshow(visual, cmap=cmap, figure=fig)
            plt.ion()
        if plt.fignum_exists('env'):
            # If you close the window, the episode will terminate.
            self._image.set_data(visual)
            self._fig.canvas.draw_idle()
            plt.pause(1.0/1000.0)
        else:
            del self._fig
            del self._image
            self.done = True
        del visual

    def handcraft_actions(self) -> Dict[int, np.ndarray]:
        """
        Handcrafted "Good Enough" algorithm to expediently solve PredatorPrey...

        For cartesian moves only, typically speaking, though the principle
            (assign by steps_to_capture) remains consistent even for "weird" moves
            (e.g. knight-move predators)
        :return: "Good Enough" actions.
        """
        if self.prey_alive == 0:
            return {p: np.random.randint(len(self.directions), size=1) for p in self.agent_id_list}
        # N = necessary predators for capture
        n = self.capture_requirement
        ### Sort prey by [total manhat dist to the nearest N preds / steps_to_capture] ascending
        living_prey = self.prey[(self.prey >= 0).all(axis=1), :]
        # [prey, pred]
        pdist = np.abs(self.predators[None, :, :] - living_prey[:, None]).sum(axis=2)
        # [prey, pred_id] Find the nearest N predators to each prey.
        # (could speed up but it's not a chokepoint)
        sorted_pred_ids = np.argsort(pdist, axis=1)
        # [prey]
        steps_to_capture = np.sort(pdist, axis=1)[:, :n].max(axis=1)
        prey_close: np.ndarray = np.argsort(steps_to_capture)
        ### For each prey in that list, in order, assign the predators to the prey.
        # Do not re-assign predators.
        action_map = {}
        filled_locations = set()
        # [pred/prey, dirs, dims]
        prey_adjs = living_prey[:, None] + self.directions[None, :, :]
        pred_adjs = self.predators[:, None, :] + self.directions[None, :, :]

        assignments = OrderedDict()
        for prey in prey_close:
            for predator in sorted_pred_ids[prey][:n]:
                if predator not in assignments:
                    assignments[predator] = prey
        # Slow, so what, it works, that's all we need for now
        for prey in prey_close:
            for predator in sorted_pred_ids[prey]:
                if predator not in assignments:
                    assignments[predator] = prey

        for predator, prey in assignments.items():
            # Move predators towards the nearest open adjacency of their assigned prey,
            next_pos = None
            curr_min_dist = sum(self.map_shape)
            for prey_adj in prey_adjs[prey]:
                if np.any((prey_adj >= self.map_shape) | (prey_adj < 0)) or \
                        self._coord_hash(prey_adj) in filled_locations:
                    continue
                # Consider distance
                next_dist = np.sqrt(np.sum(np.square(pred_adjs[predator] - prey_adj[None, :]), axis=1))
                for cand_action, cand_dist in enumerate(next_dist):
                    cand_pos = pred_adjs[predator, cand_action]
                    if np.all(np.equal(cand_pos, living_prey[prey])):
                        cand_dist = max(cand_dist, 1)
                        cand_pos = self.predators[predator]
                    if cand_dist < curr_min_dist:
                        if self._coord_hash(cand_pos) not in filled_locations:
                            action_map[predator] = np.array([cand_action])
                            curr_min_dist = cand_dist
                            next_pos = cand_pos
            if next_pos is not None:
                filled_locations.add(self._coord_hash(next_pos))

        for p_id in self.agent_id_list:
            if p_id not in action_map:
                this_pred = self.predators[p_id]
                # [dirs, dims] -> coords
                next_pos = this_pred[None] + self.directions
                # [prey, dirs] -> eucl dist to prey given direction
                peucl = np.sqrt(np.square(living_prey[:, None] - next_pos[None]).sum(axis=2))
                # [dirs] -> eucl dist to closest prey given direction
                pdist = peucl.min(axis=0)
                # Penalize exceeding bounds
                pdist[np.any(next_pos >= self.map_shape) | np.any(next_pos < 0)] += sum(self.map_shape)
                action_map[p_id] = np.array([np.argmin(pdist)])
        return action_map

    def transfer_domain(self, message: DomainTransferMessage):
        self.last_domain_transfer = message
        deletes = [k for k, v in message.agent_remapping.items() if v is None]
        for to_delete in deletes:
            self.map[tuple(self.predators[to_delete])] = 0

        remapped_ids = message.remap_id_list(self.agent_id_list)
        new_agent_ids = []
        new_agent_class_map = {}
        new_config_predators = []
        for canonical, remapped in enumerate(remapped_ids):
            new_agent_ids.append(canonical)
            new_agent_class_map[canonical] = self._agent_class_map[remapped]
            new_config_predators.append(self.config_predators[remapped])

        self._agent_class_map = new_agent_class_map
        self.config_predators = new_config_predators
        self.predators = self.predators[remapped_ids]
        self.predators_next = self.predators_next[remapped_ids]
        self._agent_id_list = new_agent_ids
        # TODO predator add case

        # TODO somewhat ironically, prey are even HARDER to add because the domains themselves are changed
        # (this definitely is something we need to address in the future)
