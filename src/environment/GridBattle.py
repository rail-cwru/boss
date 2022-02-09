"""
Multiagent "Battle" Environment.

Replicate Battle environment from MAgent.
    (https://github.com/geek-ai/MAgent)

Agents move on a discrete grid.
There are multiple teams of agents.
Each team defines configuration for its agents.
    These configs are defined in _is_team_desc.

For now, we define teams strictly and team 1 is AI agents while team 2 is preloaded enemies.
    In the future (TODO) we could modify to allow multiple teams all potentially controlled by
    possibly, actively learning AI.

Other config:
    [move_order] - a list of order for agent movement or "concurrent" for concurrent order.
                   Under concurrent order, conflicting moves block each other (i.e. do not pass through.)
    [attack_reward_matrix] -  A matrix describing rewards for attacking agents. (Shaped)
        For ROW (axis 0) observer and COLUMN (axis 1) attacked,
            the matrix holds the reward for an agent on the observer team experienced when
                         directly attacking an agent on the attacked team.
    [defeat_reward_matrix] - attack_reward_matrix but for when the agent is defeated and removed.
        If agents from different teams all contribute to a defeat, all involved teams receive their own
            rewards.

Agents at any timestep have several actions:
    Move in any direction, including no-op
    Attack in any direction, excluding no-op.

Reward rules:
    At a given timestep there is [time_reward].

Update rules:
    Enemy will operate as the second agentclass, and have a higher move range.
    If an enemy is attacked more than [enemy_life], the enemy will be removed from the field.
    If an agent is attacked more than [agent_life], the agent will be removed from the field.
    At the resolution of the turn, recover [time_recover]

TODO:
    Modify so that we can have an arbitrary amount of "teams," such that all entities
        are agents unto themselves optionally. This would be ideally conducted by having
        agentsystem as a config item of environment.

TODO:
    Learning problems:
        Agents like to block each other's movement to the same square, for some reason.
            Remedy: Competitive square-racing?
        Cannot learn good policy to check deletion on.
            Remedy: Run prelim episode which terminates instead of domain_transfer.
                    Then we can do the heuristics on that.


"""
from functools import partial
from typing import Tuple, List, Dict, Union, Callable, TYPE_CHECKING, Any, Type, Set

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from common.aux_env_info import AuxiliaryEnvInfo
from common.domain_transfer import DomainTransferMessage
from config import checks, gen_check
from config.config import ConfigItemDesc
from domain.actions import DiscreteAction
from . import Environment
from common.properties import Properties
from domain.DiscreteActionFeatureDomain import DiscreteActionFeatureDomain
from domain.observation import ObservationDomain
from domain.DiscreteActionDomain import DiscreteActionDomain
from domain.ActionDomain import ActionDomain
from domain.features import Feature, BinaryFeature, VectorFeature, NormalizedRealFeature
from common.vecmath import vec_choice, CachedOps, nd_gather, norm_l1_ball

if TYPE_CHECKING:
    from config import Config


def _random_or_positions_in_map(l):
    ret = isinstance(l, list) and len(l) > 0
    for dim in l:
        assert dim == 'random' or all([isinstance(coord, int) and coord >= 0 for coord in dim])
    return ret


def _gen_team_desc(is_agents) -> Callable:
    def _is_team_desc(data):
        assert 'init_positions' in data and _random_or_positions_in_map(data['init_positions']),\
            'A list of int coordinates or "random" describing agent positions in ID order'
        assert 'life' in data and checks.positive_float(data['life']),\
            'Team desc requires positive float "life" field.'
        assert 'recover' in data and checks.numeric(data['recover']),\
            'Team desc requires nonnegative float "recover" field.'
        assert 'attack_power' in data and checks.nonnegative_float(data['attack_power']),\
            'Amount (float) of life points an attack from an agent of this team decreases.'
        assert 'view_range' in data and checks.positive_integer(data['view_range']), \
            'View range (in squares, manhattan distance) of agents in this team.'
        assert 'attack_range' in data and checks.positive_integer(data['attack_range']),\
            'Attack range (in squares, manhattan distance) of agents in this team.'
        assert 'move_range' in data and checks.positive_integer(data['move_range']), \
            'Move range (in squares, manhattan distance) of agents in this team.'
        assert 'time_reward' in data and checks.numeric(data['time_reward']), \
            'Reward experienced for team (in total) over time.'
        assert 'attacked_reward' in data and checks.numeric(data['attacked_reward']), \
            'Reward multiplier experienced (usually negative) for being attacked. Scales with damage.'
        assert 'defeated_reward' in data and checks.numeric(data['defeated_reward']), \
            'Reward (usually negative) when a single agent is defeated.'
        assert 'collide_reward' in data and checks.numeric(data['collide_reward']), \
            'Reward (usually negative) when a agent moves into a collision.'
        assert 'miss_reward' in data and checks.numeric(data['miss_reward']), \
            'Reward (usually negative) when a agent attacks nothing.'
        assert 'system_reward' in data and isinstance(data['system_reward'], bool),\
            'Whether the reward for this system is shared among all agents in the system.' \
            'Required if using Coordination Graph Learning.'
        if not is_agents:
            assert 'ai' in data and callable(_TeamDesc._team_ai(data['ai'])),\
                'The AI used for this team (requires AI!). May be a rule-based method such as ' \
                '"stochastic_naive". In the future, other AgentSystems themselves could be used.'
        return True
    return _is_team_desc


def _list_of_teamdescs(x):
    assert len(x) >= 1
    # TODO change me when multiple AI teams supported
    ret = _gen_team_desc(True)(x[0])
    for teamdesc in x[1:]:
        ret &= _gen_team_desc(False)(teamdesc)
    return ret


class GridBattle(Environment):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc('map_shape', partial(gen_check.n_list_func, 2, checks.positive_integer),
                           info='A list of positive ints describing the shape of the map rectangle.'),
            # TODO update when config descs more expressive
            ConfigItemDesc('teams', _list_of_teamdescs, info='List of team descriptions.'),
            ConfigItemDesc('move_order', checks.positive_int_list,
                           info='Move order of teams. Either a list or "concurrent".'),
            ConfigItemDesc('attack_reward_matrix', checks.square_float_matrix,
                           info='Shaped reward experienced by ROW_TEAM for attacking COL_TEAM.'),
            ConfigItemDesc('defeat_reward_matrix', checks.square_float_matrix,
                           info='Reward experienced by ROW_TEAM for immediately'
                                'contributing to defeat of a COL_TEAM agent'),
            ConfigItemDesc('prolong_agents_on_transfer', lambda x: isinstance(x, bool),
                           info='Recover the learning team life when one of its agents is deleted.',
                           optional=True, default=False),
            ConfigItemDesc('end_on_transfer', lambda b: isinstance(b, bool),
                           info='End the episode when any system agent is deleted instead of using domain transfer.')
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=True,
                          use_agent_deletion=True,
                          use_agent_addition=True)

    @property
    def agent_class_list(self) -> List[Any]:
        return self._team_list

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    def __init__(self, config: 'Config'):
        cfg = config.find_config_for_instance(self)
        self.move_order = cfg.move_order
        self.attack_reward_matrix = cfg.attack_reward_matrix
        self.defeat_reward_matrix = cfg.defeat_reward_matrix
        self.map_shape: np.ndarray = cfg.map_shape
        self.prolong_on_defeat: bool = cfg.prolong_agents_on_transfer
        self.end_on_transfer: bool = cfg.end_on_transfer
        self.dims = len(self.map_shape)

        self.teams = []
        self._team_list = []
        n_teams = len(cfg.teams)
        for team_id, team_data in enumerate(cfg.teams):
            self._team_list.append(team_id)
            team = _TeamDesc(team_id, team_data, self.dims, n_teams,
                             self.attack_reward_matrix[team_id],
                             self.defeat_reward_matrix[team_id])
            self.teams.append(team)

        # Allocate variables for [set init state]
        self.last_display = {}
        self._agent_id_list = []
        self._agent_class_map = {}
        self.agent_team_map = {}
        self.all_agent_id_list = []
        self.map = np.array([])   # reference array used to point to team data
        self.map_dtype = np.dtype([('team', 'i2'), ('team_aid', 'i2'), ('life', 'f4'), ('coord', 'u2', (self.dims,))])

        super().__init__(config)

        self.joint_observation_domains = self._create_joint_agent_class_observation_domains(config)

    def get_auxiliary_info(self) -> AuxiliaryEnvInfo:
        return AuxiliaryEnvInfo(joint_observation_domains=self.joint_observation_domains)

    # This env uses numpy for randomness, the numpy seed is set in the controller
    def set_initial_seed(self, seed: int):
        pass

    def get_seed_state(self):
        return []

    def set_seed_state(self, seed_state):
        pass

    def _reset_state(self, visualize: bool = False) -> Any:
        """
        Initialize / reset the environment state.
        This can be relatively costly compared to the observe or update functions.
        """
        # The map just contains "what is here?" It is an arbitrary size.
        # TODO test numpy array vs nested python list
        self.map = np.full(self.map_shape, -1, dtype=self.map_dtype)
        free_squares = list(np.indices(self.map_shape).reshape(self.dims, self.map.size).T)
        np.random.shuffle(free_squares)
        curr_agent_id = 0

        # Create agents
        self.agent_team_map = {}
        self.all_agent_id_list = []
        for team in self.teams:
            curr_agent_id = team.spawn(self.map, free_squares, curr_agent_id)
            for agent_id in team.agent_ids:
                self.agent_team_map[agent_id] = team.team_idx
            self.all_agent_id_list += team.agent_ids

        # TODO (future) extend for agent ids over all teams, not just team 0
        self.learning_teams = self.teams[0:1]
        self._agent_id_list = self.teams[0].agent_ids
        self._agent_class_map = {agent_id: 0 for agent_id in self._agent_id_list}   # agent -> team id

        return self.map, self.teams

    def _create_observation_domains(self, config) -> Dict[str, ObservationDomain]:
        return {team.team_idx: team.observation_domain for team in self.learning_teams}

    def _create_joint_agent_class_observation_domains(self, config):
        # PROHIBIT CROSS-TEAM JOINT OBSERVATIONS.
        return {(team.team_idx, team.team_idx): team.joint_intra_observation_domain for team in self.learning_teams}

    def _create_action_domains(self, config) -> Dict[str, ActionDomain]:
        return {team.team_idx: team.single_action_domain for team in self.learning_teams}

    # @profile
    def observe(self, obs_groups: List[Union[int, Tuple[int, ...]]]=None)\
            -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        ret = {}
        if len(self._agent_id_list) <= 0:
            return ret
        if obs_groups is None:
            singles = self.agent_id_list
            doubles = []
        else:
            singles = [agent_id for agent_id in obs_groups if isinstance(agent_id, int)]
            doubles = [agent_id for agent_id in obs_groups if isinstance(agent_id, tuple) and len(agent_id) > 1]
        for team in self.learning_teams:
            obs = team.observe(singles, doubles, self.teams, self.map)
            ret.update(obs)
        return ret

#     # @profile
    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        displayable_moves = {}
        all_moves = {}
        team_moves = {}
        metrics = {'damage': {}}
        for team in self.teams:
            if self.gather_metrics:
                metrics['damage'][team.team_idx] = np.copy(team.agentdata['life'])  # Just the old hp right now
            if team.ai != 'system':
                moves = team.move_algorithm(self.map, self.teams)
                team_moves[team.team_idx] = moves
                all_moves.update(moves)
            else:
                team_moves[team.team_idx] = {agent_id: actions[agent_id] for agent_id in team.agent_ids}
        reward = {agent_id: self.teams[self.agent_team_map[agent_id]].time_reward for agent_id in self.agent_id_list}
        dead_agents = set()  # dest_team, dest_team_aid, dest_aid
        if self.move_order != 'concurrent':
            for team_idx in self.move_order:
                team_reward, dead_agents_update, display_update = self.resolve_actions(team_moves[team_idx],
                                                                                       set_team=self.teams[team_idx])
                for agent_id, reward_update in team_reward.items():
                    # Shortcut to knowing if agent is learnable
                    if agent_id in self.agent_id_list:
                        reward[agent_id] += reward_update
                dead_agents.update(dead_agents_update)
                displayable_moves.update(display_update)
        else:
            reward, dead_agents_update, display_update = self.resolve_actions(all_moves)
            dead_agents.update(dead_agents_update)
            displayable_moves.update(display_update)

        self.last_display = displayable_moves

        # Recover, and Agglomerate reward for system reward teams
        for team in self.teams:
            if self.gather_metrics: # This calculates damage
                metrics['damage'][team.team_idx] = team.agentdata['life'] - metrics['damage'][team.team_idx]
            np.minimum(team.max_life, team.agentdata['life'] + team.recover, out=team.agentdata['life'])
            if team.system_reward:
                system_reward = sum([reward[agent_id] for agent_id in team.agent_ids])
                for agent_id in team.agent_ids:
                    reward[agent_id] = system_reward

        # Clean up the dead
        if len(dead_agents) > 0:
            dead_announcement = ['Team{} A{}'.format(dest_team, dead_aid) for dest_team, _, dead_aid in dead_agents]
            print('Agents died this turn:', ','.join(dead_announcement))

            targeted_teams = []
            for dest_team, _, _ in dead_agents:
                teamdesc = self.teams[dest_team]
                if self.prolong_on_defeat and teamdesc in self.learning_teams and teamdesc not in targeted_teams:
                    print('Recovered team', dest_team)
                    teamdesc.agentdata['life'] = teamdesc.max_life
                    targeted_teams.append(teamdesc)

            # Prep domain transfer if dead agents - clean up data (teams already set to "-1")
            # TODO (future) extend when teams other than team 0 are trainable
            if self.end_on_transfer:
                remap_agents = {}
                for dest_team, _, agent_id in dead_agents:
                    if self.teams[dest_team] not in self.learning_teams:
                        remap_agents[agent_id] = None
                    else:
                        self.done = True
                # print('dead signal', dead_agents, 'teams', [self.agent_team_map[sig[-1]] for sig in dead_agents])
            else:
                remap_agents = {agent_id: None for _, _, agent_id in dead_agents}
            self.transfer_domain(DomainTransferMessage(remap_agents=remap_agents))
            # Check termination condition
            # TODO (future) allow training when at least one trainable team / allow running forever / config
            self.done |= any([team.num_alive() <= 0 for team in self.learning_teams])
            self.done |= all([team.num_alive() <= 0 for team in self.teams if team not in self.learning_teams])

        if self.gather_metrics:
            metrics['reward'] = reward
            if len(dead_agents) > 0:
                metrics['dead'] = dead_agents
            self.metrics.append(metrics)

        return reward

    def resolve_actions(self, actions: Dict[int, np.ndarray], set_team=None)\
            -> Tuple[Dict[int, int], Set[Tuple[int, int, int]], Dict[int, Any]]:
        """
        Perform blocking actions, update the map and positions, and return a reward dictionary.
        Don't handle time reward here, only attack / attacked / defeat / defeated / collide rewards.
        """
        # Just the reward updates
        rewards = {agent_id: 0 for agent_id in actions}
        old_positions = {}
        display_update = {}
        queued_positions = {agent_id: None for agent_id in actions}  # None: no move; else, the new coord
        tile_occupation = {}      # To implement blocking actions
        dead_agents = set()   # Dead agents
        # For each, try the action
        for agent_id, action in actions.items():
            team: _TeamDesc = set_team if set_team is not None else self.teams[self.agent_team_map[agent_id]]
            # action is the index of the direction the agent moves in
            act = action[0]
            is_move = team.move_action_mask[act]
            team_aid = team.agent_ids.index(agent_id)
            curr_data = team.agentdata[team_aid]
            old_positions[agent_id] = curr_data['coord']
            lifetext = 'T{}A{}: {:2.2f}/{:2.2f}'.format(team.team_idx, agent_id, curr_data['life'], team.max_life)
            if curr_data['life'] <= 0:   # The dead do not move
                continue
            if is_move:  # QUEUE move but don't actually move yet
                move = team.move_actions[act]
                move_to = move + curr_data['coord']
                move_to_tup = tuple(move_to)
                if np.greater_equal(move_to, self.map_shape).any() or (move_to < 0).any()\
                        or self.map[move_to_tup]['team'] >= 0: # blocked by wall or agent. short-circuit = safe check
                    queued_positions[agent_id] = None
                    rewards[agent_id] += team.collide_reward
                elif move_to_tup in tile_occupation:  # Action blocked, revert everyone involved
                    queued_positions[agent_id] = None
                    queued_positions[tile_occupation[move_to_tup]] = None
                    rewards[agent_id] += team.collide_reward
                else:  # Move was ok or we are first there before collision (and will have to revert)
                    tile_occupation[move_to_tup] = agent_id
                    queued_positions[agent_id] = (move_to, curr_data)
                display_update[agent_id] = ('move', np.copy(curr_data['coord']), move, lifetext)
            else:       # Attack action
                atk = team.attack_actions[act]
                atk_to = atk + curr_data['coord']
                atk_to_tup = tuple(atk_to)
                if np.less(atk_to, self.map_shape).all() and (atk_to >= 0).all():  # Within map
                    dest = self.map[atk_to_tup]
                    dest_team = dest['team']
                    if dest_team >= 0:  # attacks something
                        dest['life'] -= team.attack_power
                        dest_teamdesc = self.teams[dest_team]
                        dest_taid = dest['team_aid']
                        dest_aid = dest_teamdesc.agent_ids[dest_taid]
                        dest_data = dest_teamdesc.agentdata[dest_taid]
                        dest_data['life'] -= team.attack_power
                        if dest_data['life'] > 0:
                            rewards[agent_id] += team.attack_rewards[dest_team]
                            recv_reward = dest_teamdesc.attacked_reward * team.attack_power
                            if dest_aid in rewards:
                                rewards[dest_aid] += recv_reward
                            else:
                                rewards[dest_aid] = recv_reward
                        else:
                            rewards[agent_id] += team.defeat_rewards[dest_team]
                            if dest_aid in rewards:
                                rewards[dest_aid] += dest_teamdesc.defeated_reward
                            else:
                                rewards[dest_aid] = dest_teamdesc.defeated_reward
                            dead_agents.add((dest_team, dest_taid, dest_aid))
                    else:
                        rewards[agent_id] += team.miss_reward
                display_update[agent_id] = ('atk', curr_data['coord'], atk, lifetext)

        # Get rid of dead agents
        for dest_team, dest_taid, dest_aid in dead_agents:
            teamdesc = self.teams[dest_team]
            dest_data = teamdesc.agentdata[dest_taid]
            dest_data['team'] = -1
            self.map[tuple(dest_data['coord'])]['team'] = -1

        # Apply queued position to update team data and update map (including moving dead agents)
        for agent_id, queued_data in queued_positions.items():
            if queued_data is not None:
                next_pos, agent_data = queued_data
                # Move to next position and flush old position, then update coordinate in team data
                self.map[tuple(next_pos)] = self.map[tuple(old_positions[agent_id])]
                self.map[tuple(old_positions[agent_id])] = -1
                agent_data['coord'] = next_pos

        return rewards, dead_agents, display_update

    def transfer_domain(self, message: DomainTransferMessage):
        # Actually get rid of the "location" row from the team.
        # Remap the IDs.
        deletes = [k for k, v in message.agent_remapping.items() if v is None]
        changed_asys_agents = False

        for to_delete in deletes:
            teamdesc = self.teams[self.agent_team_map[to_delete]]
            # Restore health of team when any agent of team dies, for learning teams
            # TODO (future) generalize health restoration scheme for arbitrary episode prolonging
            # Delete from map (again) to make sure (esp if transfer domain was called from outside of update)
            data = teamdesc.agentdata[teamdesc.agent_ids.index(to_delete)]
            self.map[tuple(data['coord'])] = -1
            if teamdesc in self.learning_teams:
                changed_asys_agents = True

        remapped_ids = message.remap_id_list(self.all_agent_id_list)
        new_agent_id_list = []
        new_all_agent_id_list = []
        new_agent_class_map = {}
        new_agent_team_map = {}
        new_init_positions = {}
        new_teamdata = {}
        new_team_aids = {}

        # TODO fix situation where all trainable agent IDs must precede non-trainable agent IDs but not enforced
        for canonical, remapped in enumerate(remapped_ids):
            new_all_agent_id_list.append(canonical)

            team_id = self.agent_team_map[remapped]
            new_agent_team_map[canonical] = team_id
            # Gather info
            changed_asys_agents |= canonical != remapped
            teamdesc: _TeamDesc = self.teams[team_id]
            old_team_aid = teamdesc.agent_ids.index(remapped)  # New team-agentid (team internal)
            a_data: np.ndarray = teamdesc.agentdata[old_team_aid]
            # Remap in env base
            if remapped in self._agent_id_list:
                new_agent_class_map[canonical] = self._agent_class_map[remapped]
                new_agent_id_list.append(canonical)
            # Remap in team
            if teamdesc not in new_teamdata:
                new_teamdata[teamdesc] = [a_data]
                new_team_aids[teamdesc] = [canonical]
                new_team_taid = 0
                # TODO figure out better way to handle whether agent respawns at episode start
                if teamdesc in self.learning_teams:
                    # CHANGES INITIALIZATION AGENT COUNT
                    new_init_positions[teamdesc] = [teamdesc.init_positions[old_team_aid]]
            else:
                new_teamdata[teamdesc].append(a_data)
                new_team_aids[teamdesc].append(canonical)
                new_team_taid = len(new_team_aids[teamdesc]) - 1
                # (See above)
                if teamdesc in self.learning_teams:
                    new_init_positions[teamdesc].append(teamdesc.init_positions[old_team_aid])
            # Remap in map
            self.map[tuple(a_data['coord'])]['team_aid'] = new_team_taid
        # Apply new data
        for teamdesc in new_teamdata:
            teamdesc.agentdata = np.stack(new_teamdata[teamdesc], axis=0)
            teamdesc.agent_ids = new_team_aids[teamdesc]
            # (see above)
            if teamdesc in self.learning_teams:
                # print('assign {} to {}'.format(new_init_positions[teamdesc], teamdesc.init_positions))
                teamdesc.init_positions = new_init_positions[teamdesc]
        if changed_asys_agents:
            # TODO will have to change if some non-trainable agents precede trainable agents
            fixed_remap = {k: v for k, v in message.agent_remapping.items() if k in self._agent_id_list}
            self.last_domain_transfer = DomainTransferMessage(remap_agents=fixed_remap)
        self._agent_class_map = new_agent_class_map
        self._agent_id_list = new_agent_id_list
        self.agent_team_map = new_agent_team_map
        self.all_agent_id_list = new_all_agent_id_list

    def visualize(self):
        # TODO hacky - fix in future
        assert self.dims == 2
        visual = np.copy(self.map['team'] + 1)
        cmap = ListedColormap(colors=['w', 'r', 'b', 'c', 'violet', 'g'])
        bounds = BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)
        scaling = 12.0 / max(visual.shape)
        lw = max(scaling * 0.4, 0.5)
        if not hasattr(self, 'fig'):
            self.fig = plt.figure('env', figsize=(3, 3))
            plt.ion()
        if not hasattr(self, '_image'):
            fig = plt.figure('env', figsize=(3, 3))
            self._fig = fig
            self._image = plt.imshow(visual, interpolation='nearest', norm=bounds, cmap=cmap, figure=fig)
            self._arrows = {}
            # plt.subplots_adjust(right=visual.shape[1] * 0.3)
            sidetexts = []
            for k, v in self.last_display.items():
                plt.figure('env')
                acttype, arrowbase, arrowoffs, lifetext = v
                sidetexts.append(lifetext)
                hw = (1.0 if acttype != 'atk' else 1.5) * max(scaling * 0.1, 0.3)
                fc = 'k' if acttype != 'atk' else 'r'
                if np.not_equal(arrowoffs, 0).any():
                    arrow = plt.arrow(*arrowbase[::-1], *arrowoffs[::-1], fc=fc, head_width=hw, linewidth=lw, ec='k',
                                      length_includes_head=True, figure=self._fig)
                    self._arrows[k] = arrow
                else:
                    self._arrows[k] = None
            self._healthtext = plt.text(visual.shape[1] * 1.3, visual.shape[0] // 2, '\n'.join(sidetexts),
                                        fontsize=10, ha='center', va='center', figure=self._fig)
            plt.ion()
        if plt.fignum_exists('env'):
            # If you close the window, the episode will terminate.
            self._image.set_data(visual)
            sidetexts = []
            to_delete = []
            for k in self._arrows:
                if k not in self.last_display:
                    if self._arrows[k] is not None:
                        self._arrows[k].remove()
                    to_delete.append(k)
            for k in to_delete:
                del self._arrows[k]
            for k, v in self.last_display.items():
                if k in self._arrows:
                    if self._arrows[k] is not None:
                        self._arrows[k].remove()
                acttype, arrowbase, arrowoffs, lifetext = v
                sidetexts.append(lifetext)
                hw = 0.2 if acttype != 'atk' else 0.5
                fc = 'k' if acttype != 'atk' else 'r'
                if np.not_equal(arrowoffs, 0).any():
                    arrow = plt.arrow(*arrowbase[::-1], *arrowoffs[::-1], fc=fc, head_width=hw, linewidth=lw, ec='k',
                                      length_includes_head=True, figure=self._fig)
                    self._arrows[k] = arrow
                else:
                    self._arrows[k] = None
            self._healthtext.set_text('\n'.join(sidetexts))
            self._fig.canvas.draw_idle()
            plt.pause(1.0/1000.0)
        else:
            del self._fig
            del self._image
            del self._arrows
            del self._healthtext
            self.done = True
        del visual


class _TeamDesc(object):
    # Team description. Made into a class to make access more sensible.

    @classmethod
    def _team_ai(cls, method, self=None):
        src: Type['_TeamDesc'] = cls if self is None else self
        return {
            'stochastic_naive': src.__stochastic_naive,
        }[method]

    def __init__(self, team_id: int, data: Dict[str, Any], dims: int, n_teams: int,
                 attack_rewards: List[float], defeat_rewards: List[float]):
        self.team_idx = team_id
        self.n_teams = n_teams
        self.dims = dims
        self.dtype = np.dtype([('team', 'i2'), ('life', 'f4'), ('coord', 'u2', (self.dims,))])

        self.init_positions: List[Tuple[int, ...]] = data['init_positions']

        self.max_life: float = data['life']
        self.recover: float = data['recover']
        self.attack_power: float = data['attack_power']
        self.attack_rewards: np.ndarray = np.array(attack_rewards)
        self.defeat_rewards: np.ndarray = np.array(defeat_rewards)
        self.attacked_reward: float = data['attacked_reward']
        self.defeated_reward: float = data['defeated_reward']

        # Lock-covered properties
        self.__locked = False
        self.move_actions: np.ndarray
        self.move_action_mask: np.ndarray
        self.attack_actions: np.ndarray
        self.attack_action_mask: np.ndarray
        self.act_offsets: np.ndarray
        self.attack_range: float = data['attack_range']
        self.view_range: float = data['view_range']
        self.move_range: float = data['move_range']
        self.__unlock_regen_actions()

        # Declarations
        self.obs_size: int
        self.view_offsets: np.ndarray
        self.attacks: np.ndarray
        self.attacks2: np.ndarray
        self.moves: np.ndarray
        self.moves2: np.ndarray

        self.time_reward: float = data['time_reward']
        self.collide_reward: float = data['collide_reward']
        self.miss_reward: float = data['miss_reward']
        self.system_reward: bool = data['system_reward']

        self.ai: Any = 'system' if 'ai' not in data else data['ai']
        if self.ai != 'system':
            self.move_algorithm = _TeamDesc._team_ai(self.ai, self)

        # Agent data
        self.agentdata: np.ndarray = np.array([])  # Allocate for init
        self.agent_ids = []

        # Learning data
        self.single_action_domain = self.get_action_domain()
        self.joint_action_domain = DiscreteActionDomain(self.single_action_domain.items * 2, num_agents=2)
        self.observation_domain = self.get_observation_domain()
        self.joint_intra_observation_domain = self.get_joint_observation_domain()

    @property
    def attack_range(self) -> float:
        return self.__attack_range

    @attack_range.setter
    def attack_range(self, attack_range):
        assert not self.__locked
        self.__attack_range = attack_range
        self.attacks = norm_l1_ball(attack_range, self.dims)
        self.attacks2 = np.stack(DiscreteActionFeatureDomain.broadcast_joint(self.attacks, self.attacks), axis=1)

    @property
    def view_range(self) -> float:
        return self.__view_range

    @view_range.setter
    def view_range(self, view_range):
        assert not self.__locked
        self.__view_range = view_range
        self.view_offsets = norm_l1_ball(view_range, self.dims)
        self.obs_size = self.view_offsets.shape[0]

    @property
    def move_range(self):
        return self.__move_range

    @move_range.setter
    def move_range(self, move_range):
        assert not self.__locked
        self.__move_range = move_range
        self.moves = norm_l1_ball(move_range, self.dims)
        self.moves2 = np.stack(DiscreteActionFeatureDomain.broadcast_joint(self.moves, self.moves), axis=1)

    def __unlock_regen_actions(self):
        self.move_actions = np.concatenate((self.moves, np.zeros_like(self.attacks)), axis=0)
        self.move_action_mask = np.concatenate((np.full(self.moves.shape[0], True, dtype=bool),
                                                np.full(self.attacks.shape[0], False, dtype=bool)), axis=0)
        self.attack_actions = np.concatenate((np.zeros_like(self.moves), self.attacks), axis=0)
        self.attack_action_mask = ~self.move_action_mask
        self.act_offsets = np.concatenate((self.moves, self.attacks), axis=0)
        self.__locked = False

    def num_alive(self):
        return (self.agentdata['team'] == self.team_idx).sum()

    def spawn(self, env_map: np.ndarray, free_squares: List[np.ndarray], curr_agent_id: int) -> int:
        """
        Spawn the agents in the map starting from the curr_agent_id. Use free_squares to pop free squares.
        :return: The next free agent id.
        """
        self.agent_ids = []
        self.agentdata = np.zeros(len(self.init_positions), dtype=self.dtype)
        for i, position in enumerate(self.init_positions):
            if not isinstance(position, list):  # 'random', else already coord
                try:
                    picked = None
                    while picked is None:
                        picked = tuple(free_squares.pop())
                        if env_map[picked]['team'] >= 0:
                            picked = None
                    position = picked
                except IndexError:
                    raise ValueError('Not enough squares to spawn all agents. Please inspect the initial positions '
                                     'for agents and make sure the map size is large enough, or that the number of '
                                     'agents is small enough.')
            try:
                actor_slot = self.agentdata[i]
                actor_slot['team'] = self.team_idx
                actor_slot['life'] = self.max_life
                actor_slot['coord'] = position
                map_slot = env_map[tuple(position)]
                map_slot['team'] = self.team_idx
                map_slot['team_aid'] = i
                map_slot['life'] = self.max_life
                map_slot['coord'] = position
                self.agent_ids.append(curr_agent_id + i)
            except IndexError as e:
                print(e)
                raise ValueError('Agent of team {} at position [{}] was out of bounds in map of shape [{}]. '
                                 'Please check the environment configuration before running the experiment.'
                                 .format(self.team_idx, position, env_map.shape))
        return len(self.init_positions) + curr_agent_id

    def get_action_domain(self) -> DiscreteActionDomain:
        # Move and attack actions.
        num_actions = len(self.moves) + len(self.attacks)
        act_move = DiscreteAction('move_attack', num_actions)
        return DiscreteActionDomain([act_move], len(self.agent_ids))

    def get_observation_domain(self) -> DiscreteActionFeatureDomain:
        # Action-features for Value Approximation.
        assert self.single_action_domain is not None
        # Move blocked / attack goes outside field
        feat_blocked = BinaryFeature(name='blocked')
        # Attack an agent, for each team; plusminus. reversed for own team. proportional to abs(atk-pwr / enemy life)
        feat_teams = VectorFeature(name='teams', ndim=self.n_teams, min=-1, max=1)
        # Miss attack (hacky - inverted binary feature)
        feat_miss = Feature(name='miss', shape=[1], dtype='int', drange=slice(-1, 1))
        # Target not own team
        feat_nonself = BinaryFeature(name='self')
        # Min of normalized square distance to closest visible agent of each team.
        distmin = VectorFeature(name='distmin_team', ndim=self.n_teams, min=0, max=1)
        obs_domain = ObservationDomain([feat_blocked, feat_teams, feat_miss, feat_nonself, distmin])
        return DiscreteActionFeatureDomain(obs_domain, self.single_action_domain)

    def get_joint_observation_domain(self) -> DiscreteActionFeatureDomain:
        # Action-tied Joint observation for intra-team.
        # normalized square distance to each other relative to map dimension
        coord_dist = NormalizedRealFeature(name='coord_dist')
        # Both do NOT target same location (blocking action)
        n_mutual_pos = BinaryFeature(name='n_mutual_pos')
        # Both attack member of [team] at the same location
        mutual_atk = BinaryFeature(name='mutual_atk')
        obs_domain = ObservationDomain([coord_dist, n_mutual_pos, mutual_atk], num_agents=2)
        return DiscreteActionFeatureDomain(obs_domain, self.joint_action_domain)

    # @profile
    def observe(self, singles, doubles, teams, prev_map) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        # TODO (future) view range does nothing now - we use action tied features instead. Maybe incorporate?
        # TODO (!?) consider ID canonicity constraints in the future (observe dead agents!?)
        curr_pos = self.agentdata['coord'][singles, None, :]  # [a, ____, dims]
        next_pos = curr_pos + self.move_actions[None, ...]  # [a, dirs, dims]
        next_atk = curr_pos + self.attack_actions[None, ...]  # [a, atks, dims]
        next_off = curr_pos + self.act_offsets[None, ...]
        others = np.concatenate([teamdesc.agentdata for teamdesc in teams if teamdesc != self], axis=0)
        single_obs = self.__observe_single(next_pos, next_atk, next_off, singles, others, prev_map)
        ret = {agent_id: single_obs[i].flatten() for i, agent_id in enumerate(singles)}
        if doubles:
            joint_obs = self.__observe_double(curr_pos, next_pos, next_atk, doubles, prev_map)
            for i, pair in enumerate(doubles):
                ret[pair] = joint_obs[i]
        return ret

    # @profile
    def __observe_single(self, next_pos: np.ndarray, next_atk: np.ndarray, next_off: np.ndarray,
                         singles, others: np.ndarray, prev_map: np.ndarray):
        o = np.zeros((len(singles), *self.observation_domain.packed_shape), dtype='float32')

        ### "FAIR" BUT DIFFICULT TO LEARN BASIS?
        # Moves only: [a. dirs]
        valid_move_mask = (next_pos >= 0).all(axis=-1) & (next_pos < prev_map.shape).all(axis=-1)
        valid_move = next_pos[valid_move_mask]

        # Map slots the action moves this agent to [a, dirs][team, team_aid, coord]
        move_map = np.full((len(singles), next_pos.shape[1]), -1, dtype=prev_map.dtype)  # Default empty data
        move_map[valid_move_mask] = nd_gather(prev_map, valid_move)
        move_hits = move_map['team'] >= 0       # ...and where there is an agent [a, dirs]

        # Attacks which are within bounds [a, dirs]
        valid_atk_mask = (next_atk >= 0).all(axis=-1) & (next_atk < prev_map.shape).all(axis=-1)
        valid_atk = next_atk[valid_atk_mask]
        atk_map = np.full((len(singles), next_atk.shape[1]), -1, dtype=prev_map.dtype)  # Default empty data
        atk_map[valid_atk_mask] = nd_gather(prev_map, valid_atk)

        # Get feat_blocked (moves blocked, binary) [a, dirs](1)
        feat_blocked = np.copy(~valid_move_mask)
        feat_blocked[~move_hits] = False
        # Not moving (attacking) doesn't block movement, but we should get rid of out-of-bound attacks
        feat_blocked[:, self.attack_action_mask] = ~valid_atk_mask[:, self.attack_action_mask]

        # Get feat_teams (attack team, plusminus) [a, dirs, n_teams]
        # Scale according to impact of attack
        alpha = 0.3
        impact = np.abs(self.attack_power / np.maximum(atk_map['life'], self.attack_power)) * alpha + (1-alpha)
        feat_teams = np.stack([((atk_map['team'] == team) * impact).reshape(len(singles), -1)
                               for team in range(self.n_teams)], axis=2)
        # Not moving doesn't mean you attack yourself
        feat_teams[..., self.team_idx] = 1 - feat_teams[..., self.team_idx]
        feat_teams = (feat_teams * 2) - 1
        feat_teams[:, self.move_action_mask] = 0

        feat_miss = (atk_map['team'] == -1)[..., None] * -1

        # Actions which are within bounds [a, dirs]
        valid_tgt_mask = (next_off >= 0).all(axis=-1) & (next_off < prev_map.shape).all(axis=-1)
        valid_tgt = next_off[valid_tgt_mask]
        tgt_map = np.full((len(singles), next_off.shape[1]), -1, dtype=prev_map.dtype)  # Default empty data
        tgt_map[valid_tgt_mask] = nd_gather(prev_map, valid_tgt)

        feat_nonself = (tgt_map['team'] != self.team_idx)[..., None]

        # Visible agents that are not yourself
        # Modified distance; [__, __, others, dims] - [a, dirs, __, dims] -> [a, dirs, others] Higher when closer.
        # TODO (likely bug spot)
        targets = others[np.abs(others['coord']).sum(axis=-1) <= self.view_range]
        # targets = move_map[move_hits]  # [others, dims] - doesn't work for some reason. kept commented for legacy.
        dist = np.square(targets['coord'][None, None] - next_pos[:, :, None]).sum(axis=-1)
        dist = np.maximum(dist - self.attack_range, 0)

        mdd = 1 - (dist/np.square(self.move_range))
        mdd = np.clip(np.exp(mdd) / np.e, a_min=-1, a_max=1)
        # Get distavg, distmin (float) [a, dirs, n_teams] based on move
        feat_distmin = []
        for team in range(self.n_teams):
            cands = mdd[:, :, targets['team'] == team]  # [a, dirs, others]
            distmin = cands.min(axis=2) if cands.size > 0 else np.zeros(mdd.shape[:2], dtype='float32')
            feat_distmin.append(distmin)
        feat_distmin = np.stack(feat_distmin, axis=2)

        o[:] = np.concatenate([feat_blocked[:, :, None], feat_teams, feat_miss, feat_nonself,
                               feat_distmin
                               ], axis=2)

        ##### OVER-ENGINEERED VERY NONLINEAR "CHEATING" BASIS?
        # # If cannot attack the enemy, softmax by "square with expected reward for attacking"
        # num_agents = len(singles)
        # num_actions = self.act_offsets.shape[0]
        # max_dist = np.linalg.norm(prev_map.shape, ord=2)
        # prob = np.zeros((num_agents, num_actions), dtype='float32')
        #
        # # Target location: [agent, act, dims]
        # targets = self.agentdata['coord'][:, None, :] + self.act_offsets[None]
        # act_in_bounds_mask = (targets >= 0).all(axis=2) & (targets < prev_map.shape).all(axis=2)
        # atk_rews: np.ndarray = self.attack_rewards[others['team']][None, None]
        #
        # # d([a, act, __, dims], [__, __, other, dims]) -> [a, act, other]:normalized L2-dist
        # next_dists = np.linalg.norm(targets[:, :, None] - others['coord'][None, None], ord=2, axis=3)
        # # What we really want is "distance to being within range of attacking enemy" ; flattens out at 0
        # next_move_dists = next_dists[:, self.move_action_mask]
        # next_move_dists[next_move_dists <= 0] = max_dist  # Avoid moving to blocked locations
        # np.maximum(next_move_dists - self.attack_range, 0, out=next_move_dists)
        # mv_prob = 1 - (next_move_dists / (max_dist - self.attack_range))
        # # Scale by the reward for attacking the other agent if the move gets us within range
        # mv_prob *= atk_rews  # [a, act, other] : expected closeness reward to each enemy agent
        # prob[:, self.move_action_mask] = mv_prob.max(axis=2)
        #
        # # Find if possible to attack enemy; Reward for attacking square, zero if nothing, nonzero if agent
        #
        # # Weigh stochastically by expected reward * min dist to agent   [other][...]
        # other_life: np.ndarray = others['life']
        #
        # atk_hit_rews = (next_dists[:, self.attack_action_mask] == 0) * atk_rews  # [agent, act, other]:exp. atk reward
        # atk_hit_max_rew = atk_hit_rews.max(axis=2)  # [agent, atk-act]:best attack reward for making attack
        # good_hits = atk_hit_max_rew > 0  # [agent, atk-act]:positive where (agent does atk) is good hit
        # num_good_hits = good_hits.sum(axis=1)  # [agent] num of good hits available to each agent
        #
        # # For those agents, don't do anything else
        # best_hit_agents = atk_hit_rews.argmax(axis=2)  # [agent, atk-act]:agentid of target
        # best_hit_life: np.ndarray = other_life[best_hit_agents]  # [agent, atk-act]:life of target
        # best_hit_life[~good_hits] = best_hit_life.max() + 1  # Eliminate picking missed actions
        # best_hit_mask = best_hit_life == best_hit_life.min(axis=1, keepdims=True)  # [agent, atk-act]:act hits minlife
        # # Make hitting the best-yielding agent with minlife chosen
        # # TODO inconsistent in weird case where sign of defeat reward is inconsistent with attack reward.
        # for a, has_good_hit in enumerate(num_good_hits >= 1):
        #     if has_good_hit:
        #         prob[a, self.move_action_mask] *= 0.0  # TODO configurable aggressiveness
        #         prob[a, self.attack_action_mask] = best_hit_mask[a]
        #
        # # Avoid invalid actions
        # prob[~act_in_bounds_mask] = 0
        # o[..., :] = prob[..., None]
        return o

    # @profile
    def __observe_double(self, curr_pos, next_pos, next_atk, doubles, prev_map):
        o = np.zeros((len(doubles), *self.joint_intra_observation_domain.packed_shape), dtype='float32')
        max_dist = np.linalg.norm(prev_map.shape, ord=2)
        next_jpos = next_pos[np.array(doubles)]
        next_jatk = next_atk[np.array(doubles)]
        # pair, dirs, a1a2, dims
        next_jpos = CachedOps.broadcast_joint_3d_stack2(next_jpos[:, 0], next_jpos[:, 1])
        next_jatk = CachedOps.broadcast_joint_3d_stack2(next_jatk[:, 0], next_jatk[:, 1])
        # Get coord_dist (float) [pair, dirs]
        next_eucl_dist = np.linalg.norm(next_jpos[:, :, 0] - next_jpos[:, :, 1], ord=2, axis=-1)
        # Blocking actions...
        curr_jpos = curr_pos[np.array(doubles)] # pair, a1a2, _, dims
        curr_dist = np.linalg.norm(curr_jpos[:, 0, 0] - curr_jpos[:, 1, 0], ord=2, axis=1, keepdims=True) # pair, _
        blocked_mask = next_eucl_dist == 0
        for pair, pair_blocked_mask in enumerate(blocked_mask):
            next_eucl_dist[:, pair_blocked_mask] = curr_dist[pair]
        # Only want to move within helping range
        next_eucl_dist = np.maximum(next_eucl_dist - self.attack_range // 2, 0)
        coord_dist = 1 - np.log(1+np.maximum(next_eucl_dist, 1) / max_dist)
        # Get n_mutual_pos, mutual_atk (int), [pair, dirs]
        not_mutual_pos = ~blocked_mask

        # [pair, dirs] attacks same square; also make sure it is not a joint MISSED attack (awful)
        mutual_atk = np.equal(next_jatk[:, :, 0], next_jatk[:, :, 1]).all(axis=2)
        for pair, mutuals in enumerate(mutual_atk):
            # Agreed-upon joint attacks: [v-dirs, dims]
            next_a_jatk: np.ndarray = next_jatk[pair, mutuals, 0]
            valid_atk_mask = (next_a_jatk >= 0).all(axis=-1) & (next_a_jatk < prev_map.shape).all(axis=-1)
            valid_atk = next_a_jatk[valid_atk_mask].astype('int32')
            # Prepare to gather the targets of the attacks (any team)
            atk_map = np.full(next_a_jatk.shape[0], -1, dtype=prev_map.dtype)
            atk_map[valid_atk_mask] = nd_gather(prev_map, valid_atk)
            mutual_atk[pair, mutuals] *= atk_map['team'] >= 0

        o[:] = np.stack([coord_dist, not_mutual_pos, mutual_atk*0], axis=2)
        return o

    def move_algorithm(self, curr_map, all_teams) -> Dict[int, np.ndarray]:
        raise NotImplementedError('Function should be overwritten during init time via ai config.')

    def __stochastic_naive(self, curr_map: np.ndarray, all_teams: List['_TeamDesc']) -> Dict[int, np.ndarray]:
        """
        Heuristic action algorithm
        """

        # If cannot attack the enemy, softmax by "square with expected reward for attacking"
        num_agents = len(self.agent_ids)
        num_actions = self.act_offsets.shape[0]
        max_dist = np.linalg.norm(curr_map.shape, ord=2)
        prob = np.zeros((num_agents, num_actions), dtype='float32')

        # Target location: [agent, act, dims]
        targets = self.agentdata['coord'][:, None, :] + self.act_offsets[None]
        act_in_bounds_mask = (targets >= 0).all(axis=2) & (targets < curr_map.shape).all(axis=2)
        valid_target = targets[act_in_bounds_mask]  # [agent, v-act, dims]

        # Weigh stochastically by expected reward * min dist to agent   [other][...]
        others = np.concatenate([teamdesc.agentdata for teamdesc in all_teams if teamdesc != self], axis=0)
        atk_rews: np.ndarray = self.attack_rewards[others['team']][None, None]
        other_life: np.ndarray = others['life']

        # d([a, act, __, dims], [__, __, other, dims]) -> [a, act, other]:normalized L2-dist
        next_dists = np.linalg.norm(targets[:, :, None] - others['coord'][None, None], ord=2, axis=3)
        # What we really want is "distance to being within range of attacking enemy" ; flattens out at 0
        next_move_dists = next_dists[:, self.move_action_mask]
        next_move_dists[next_move_dists <= 0] = max_dist  # Avoid moving to blocked locations
        np.maximum(next_move_dists - self.attack_range, 0, out=next_move_dists)
        mv_prob = 1 - (next_move_dists / (max_dist-self.attack_range))
        # Scale by the reward for attacking the other agent if the move gets us within range
        mv_prob *= atk_rews  # [a, act, other] : expected closeness reward
        prob[:, self.move_action_mask] = mv_prob.max(axis=2)

        # Find if possible to attack enemy; Reward for attacking square, zero if nothing, nonzero if agent
        atk_hit_rews = (next_dists[:, self.attack_action_mask] == 0) * atk_rews  # [agent, act, other]:exp. atk reward
        atk_hit_max_rew = atk_hit_rews.max(axis=2)  # [agent, atk-act]:best attack reward for making attack
        good_hits = atk_hit_max_rew > 0  # [agent, atk-act]:positive where (agent does atk) is good hit
        num_good_hits = good_hits.sum(axis=1)  # [agent] num of good hits available to each agent

        # For those agents, don't do anything else
        best_hit_agents = atk_hit_rews.argmax(axis=2)  # [agent, atk-act]:agentid of target
        best_hit_life: np.ndarray = other_life[best_hit_agents]  # [agent, atk-act]:life of target
        best_hit_life[~good_hits] = best_hit_life.max() + 1  # Eliminate picking missed actions
        best_hit_mask = best_hit_life == best_hit_life.min(axis=1, keepdims=True)  # [agent, atk-act]:act hits minlife
        # Make hitting the best-yielding agent with minlife chosen
        # TODO inconsistent in weird case where sign of defeat reward is inconsistent with attack reward.
        for a, has_good_hit in enumerate(num_good_hits >= 1):
            if has_good_hit:
                prob[a, self.move_action_mask] *= 0.2  # TODO configurable aggressiveness
                prob[a, self.attack_action_mask] = best_hit_mask[a]

        # Avoid invalid actions
        prob[~act_in_bounds_mask] = 0
        # Reasonably stochastic
        actions = vec_choice(np.exp(8.0 * prob), np.r_[:num_actions])  # TODO <- difficulty?
        # Best action (killer)... this is supposed to be naive, not "handcraft algorithm"
        # actions = argmax_random_tiebreak(prob, axis=1)  # TODO way too deadly
        return {agent_id: np.array([action], dtype='int32') for agent_id, action in zip(self.agent_ids, actions)}

