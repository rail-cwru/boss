"""
A Coordination Graph system.

As in https://dl.acm.org/citation.cfm?id=757784

Explicitly for Coordination Graph Q Learning. Other formulations are untested.

Only compatible with environments which provide DiscreteActionFeatureDomain as ObservationDomain
                                            and DiscreteIndexedActionDomain as ActionDomain.

Only compatible with ActionFreeLinear function approximator.

Because of the action selection techniques specifically used for CoordinationGraphSystem, there is no current support
    for differentiating exploration and exploitation in action calculation.

The LSPI and Policy Gradient (REINFORCE) algorithms defined on this agentsystem
    mandate atypical update methods and action selection algorithms which might not correspond to default Q-learning.

This agentsystem learns Q-functions on edges. Optionally, it can also learn on vertices (single agents).
"""
import random

import numpy as np
from typing import List, Dict, Tuple, Callable, Union, Any, Type
import copy

from agentsystem.util.coordination_graph import CoordinationGraph, EliminationTree
from common.aux_env_info import AuxiliaryEnvInfo
from common.trajectory import Trajectory
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc
from policy.function_approximator.ActionFreeLinearNpy import ActionFreeLinearNpy
from . import AgentSystem
from .util import PolicyGroup
from common.properties import Properties
from common.domain_transfer import DomainTransferMessage
from domain.DiscreteActionFeatureDomain import DiscreteActionFeatureDomain
from domain.observation import ObservationDomain
from domain.DiscreteActionDomain import DiscreteActionDomain
from domain.ActionDomain import ActionDomain
from config import Config, checks
from algorithm import Algorithm
from policy import Policy


NINF = -np.inf


class CoordinationGraphSystem(AgentSystem, AbstractModuleFrame):
    """
    An agent system that coordinates agents via a graph structure.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='coordination_graph',
                           check=CoordinationGraph.validate_config,
                           info='A list of pair-lists denoting pairs of agents. '
                                'The pairs of Agent IDs should always be in ascending order for consistency.'),
            ConfigItemDesc(name='selection_method',
                           check=lambda s: s in ['greedy_with_restarts', 'max_plus', 'agent_elimination'],
                           info='The selection method to use in coordination graph system. '
                                'It may be one of the following:'
                                '\n\tgreedy_with_restarts: A greedy selection that is not guaranteed to converge to '
                                'the optimal joint action.'
                                '\n\tmax_plus: The centralized Max Plus algorithm which can converge quickly to an '
                                'optimal joint action with a number of iterations, depending on the average degree '
                                'of the agent coordination graph.'
                                '\n\tagent_elimination: A more computationally intensive algorithm which is '
                                'guaranteed to converge to the optimal joint action for the coordinated agents.'),
            ConfigItemDesc(name='num_iterations',
                           check=checks.positive_integer,
                           info='Number of iterations used for the action selection algorithm (if greedy or max_plus)'),
            ConfigItemDesc(name='transfer_method',
                           check=lambda s: s in ['drop_restart', 'drop_removed',
                                                 'map_to_neighbors', 'project_over_existing'],
                           info='Transfer method for agent removal. Can be of: '
                                'drop_restart, drop_removed, map_to_neighbors, project_over_existing'),
            ConfigItemDesc(name='use_nodes',
                           check=lambda b: isinstance(b, bool),
                           info='Whether to also include node potentials.',
                           optional=True,
                           default=False)
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_coordination_graph=True,
                          use_joint_observations=True,
                          use_single_agent=False,
                          use_discrete_action_space=True)

    def __init__(self,
                 agent_id_class_map: Dict[int, int],
                 agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain],
                 auxiliary_data: AuxiliaryEnvInfo,
                 experiment_config: Config):
        """
        Expects:

        joint_agent_class_observation_domains: Map of tuples of agent class IDs to the joint observation domains
        """

        # Config settings
        self.experiment_config = experiment_config
        asys_config = experiment_config.find_config_for_instance(self)
        self.num_iterations = asys_config.num_iterations
        self.transfer_method = asys_config.transfer_method
        self.use_nodes = asys_config.use_nodes

        assert auxiliary_data.joint_observation_domains
        joint_agent_class_observation_domains = auxiliary_data.joint_observation_domains
        self.joint_agent_class_observation_domains = joint_agent_class_observation_domains

        self.graph: CoordinationGraph = CoordinationGraph(agent_id_class_map, asys_config.coordination_graph, strict=True)
        self.elimtree = EliminationTree(self.graph)

        # Joint features only.
        self._observation_sets: List[Union[int, Tuple[int, ...]]] = [agent_id for agent_id in agent_id_class_map.keys()]
        self._observation_sets += [tuple(edge) for edge in self.graph.edges]

        # Avoid regeneration of extant domains. Map joint agentclass to obs_domain and act_domain.
        self._cached_joint_domains: Dict[Tuple[Any, Any], Tuple[ObservationDomain, ActionDomain]] = {}

        # Accelerate action selection
        self._edge_q_cache = {}

        self._validate(agent_class_action_domains,
                       agent_class_observation_domains,
                       joint_agent_class_observation_domains,
                       experiment_config)

        # Create helpful action range data for action selection algorithms
        self._agent_class_action_range_map = {}
        self._agent_id_action_size_map = {}
        for agent_id, agent_class in agent_id_class_map.items():
            if agent_class not in self._agent_class_action_range_map:
                action_domain = agent_class_action_domains[agent_class]
                self._agent_class_action_range_map[agent_class] = action_domain.get_action_range()
            action_range = self._agent_class_action_range_map[agent_class]
            self._agent_id_action_size_map[agent_id] = action_range.stop - action_range.start

        # Determine action selection
        self.action_selection: Callable[[Any], Dict[int, np.ndarray]] = {
            'greedy_with_restarts': self._greedy_with_restarts,
            'agent_elimination': self._agent_elimination,
            'max_plus': self._max_plus,
        }[asys_config.selection_method]

        self._last_pg_observations = {}

        # TODO ===================
        self.top_config = experiment_config

        # TODO put policy and algorithm into asys requirements FOR SUBCLASSES
        # TODO figure out how on earth to get it organized sanely
        self.policy_class: Type[Policy] = self.top_config.policy.module_class

        # TODO - remove all references, use instantiated algorithm where possible
        # ASys owns algorithm .... should only IndepSys and SharedSys have the single alg?
        # Where along the ladder do heterogeneous algorithms / policies exist?
        # TODO heterogeneous alg only by modification?
        self.algorithm = self.top_config.algorithm.module_class(experiment_config)
        self.algorithm_class: Type[Algorithm] = self.top_config.algorithm.module_class

        # Map from CANONICAL AGENT ID (uuid) to AGENT CLASS ID
        self.agent_id_class_map: Dict[int, int] = agent_id_class_map

        # CANONICAL AGENT IDs (uuid)
        self.agent_ids: List[int] = [key for key in agent_id_class_map.keys()]
        self.agent_classes: List[int] = list(set(agent_id_class_map.values()))
        sorted(self.agent_classes)

        # Inverse map from agent class ID to list of agents
        self.agent_class_id_map: Dict[int, List[int]] = {agent_class: [agent_id for agent_id, agent_id_class
                                                                       in self.agent_id_class_map.items()
                                                                       if agent_id_class == agent_class]
                                                         for agent_class in self.agent_classes}

        # Action Domain for each Agent Class
        self.agent_class_action_domains: Dict[int, ActionDomain] = agent_class_action_domains

        # Observation Domain for each Agent Class
        self.agent_class_observation_domains: Dict[int, ObservationDomain] = agent_class_observation_domains

        # Method of production determined by concrete class
        self.max_time = experiment_config.episode_max_length  # TODO find? remove if annoying? etc.
        self.policy_groups: List[PolicyGroup] = self._make_policy_groups(
            self.agent_id_class_map,
            self.agent_class_action_domains,
            self.agent_class_observation_domains)

        # agent_id -> list[policy_groups]
        self.agent_policy_groups_map: Dict[int, List[PolicyGroup]] = {agent_id: [] for agent_id in self.agent_ids}
        for pg in self.policy_groups:
            for agent_id in pg.agents:
                self.agent_policy_groups_map[agent_id].append(pg)

        # Validate observation request output
        observation_request = self.observe_request()
        if observation_request is not None:
            for obs_key in observation_request:
                msg_err = 'Encountered an invalid agent or joint agent of the form [{}]' \
                          'for which an observation was requested.'.format(obs_key)
                assert isinstance(obs_key, int) or (isinstance(obs_key, tuple) and len(obs_key) > 1), msg_err

        self.next_pg_id = len(self.policy_groups)

    def _validate(self,
                  agent_class_action_domains,
                  agent_class_observation_domains,
                  joint_agent_class_observation_domains,
                  experiment_config):
        # Theoretic problem. CGQL only supports discrete action spaces.
        # Combinatorial problem (tractability). CGQL blows up with too many discrete actions.
        action_sizes = {}
        for agent_class, action_domain in agent_class_action_domains.items():
            assert isinstance(action_domain, DiscreteActionDomain), \
                "CoordinationGraphSystem currently only supports DiscreteIndexedActionDomain."

            # Only allow simple actions for now.
            assert len(action_domain.items) < 2, "CoordinationGraph Sys doesn't support complex actions.\n" \
                                                 "If you have multiple discrete actions in different array elements, " \
                                                 "please compact all of them to a single item in your array.\n" \
                                                 "e.g. 0: Do nothing, 1: Go up, 2: Go right, etc. instead of using " \
                                                 "literal coordinates as the action data, which would take up two " \
                                                 "spaces in the array."
            action_sizes[agent_class] = action_domain.shape[0]

        dom_msg = 'All Observation Domains must be DiscreteActionFeatureDomain' \
                  'for CoordinationGraphSystem to operate correctly.'
        for obs_domain in agent_class_observation_domains.values():
            assert isinstance(obs_domain, DiscreteActionFeatureDomain), dom_msg
        for joint_agent_class, joint_obs_domain in joint_agent_class_observation_domains.items():
            if len(joint_agent_class_observation_domains) == 2:
                assert isinstance(joint_obs_domain, DiscreteActionFeatureDomain), dom_msg
                ac1, ac2 = joint_agent_class
                is_joint_action_feature = joint_obs_domain.actions.shape[0] == (action_sizes[ac1] * action_sizes[ac2])
                assert is_joint_action_feature, 'The Joint action-features must be over the joint action of the ' \
                                                'involved agent classes.'

        # Theoretical constraint. CGS transfer is only ok with actionfree linear policy FA, unproven on others.
        if self.transfer_method != 'drop_removed':
            # TODO messy and bad - any other way to restrict?
            fa_name = experiment_config.policy.function_approximator.name
            assert fa_name in ['Linear', 'ActionFreeLinear', 'ActionFreeLinearNpy'],\
                'CoordinationGraphSystem does not support FunctionApproximator other than ActionFreeLinear or ' \
                'Linear, but [{}] was given.'.format(fa_name)

    def __get_action_range(self, agent_class):
        # THIS DOES NOT MAKE SENSE FOR COMPLEX ACTION SPACES!
        return self.agent_class_action_domains[agent_class].get_action_range()

    def _make_policy_groups(self, agent_class_map: Dict[int, int], agent_class_action_domains: List[ActionDomain],
                            agent_class_observation_domains: List[ObservationDomain]) -> List[PolicyGroup]:
        # Edges correspond to policy groups, but agents may belong to multiple edges.
        # So the PG mapping doesn't exactly make sense.
        # At the same time, using CG explicitly instead of implicitly folding it into PGs might be more proper?
        # A policy group for each edge:
        policy_groups = []
        for edge_id in self.graph.edge_ids:
            policy_group = self._make_joint_policy_group(edge_id, edge_id)
            self.graph.assign_edge_pg(edge_id, policy_group)
            policy_groups.append(policy_group)
        # If using nodes (agent PGs a la IndependentSystem), use negative numbers from -1 down to avoid conflicts w/ eid
        if self.use_nodes:
            for agent_id in agent_class_map:
                pg_id = -1 - agent_id
                obs_dom = self._get_agent_observation_domain(agent_id)
                act_dom = self._get_agent_action_domain(agent_id)
                policy = self.policy_class(obs_dom, act_dom, self.experiment_config)
                model = self.algorithm_class.make_model(policy, self.experiment_config)
                pg = PolicyGroup(pg_id, [agent_id], policy, model, self.max_time)
                self.graph.assign_vertex_pg(agent_id, pg)
                policy_groups.append(pg)
        return policy_groups

    def _make_joint_policy_group(self, edge_id: int, pg_id: int) -> PolicyGroup:
        """
        Make policy group object on an edge (only for this system)
        :param edge_id: Edge id we are making a policy group for
        :return:
        """
        policy = self._make_joint_policy(edge_id)
        model = self.algorithm_class.make_model(policy, self.experiment_config)
        policy_group = PolicyGroup(pg_id, self.graph[edge_id], policy, model, self.max_time)
        return policy_group

    def _make_joint_policy(self, edge_id: int):
        """
        Make the joint policy object for the pair of agents at the defined edge.
        :param edge_id: Edge we are making a policy for.
        :return:
        """
        joint_act_dom, joint_obs_dom = self._make_joint_domains(edge_id)
        policy = self.policy_class(joint_obs_dom, joint_act_dom, self.experiment_config)
        return policy

    def _make_joint_domains(self, edge_id):
        """
        Create joint domains for the two agents, reusing if already existant.
        :param a1: Agent 1
        :param a2: Agent 2
        :param edge_id: Edge
        :return:
        """
        joint_class = self.graph.get_edge_class(edge_id)
        a1_class, a2_class = joint_class

        # By necessity, it is somewhat different from the InverseRL CGQL. We append the independent features
        # And the joint observation feature together to create the edge.
        a1_obs = self.agent_class_observation_domains[a1_class]
        a2_obs = self.agent_class_observation_domains[a2_class]

        if joint_class in self._cached_joint_domains:
            joint_obs_dom, joint_act_dom = self._cached_joint_domains[joint_class]
        else:
            joint_observation_domain = self.joint_agent_class_observation_domains[joint_class]
            joint_obs_dom = DiscreteActionFeatureDomain.join(a1_obs, a2_obs, joint_observation_domain)
            a1_act = self.agent_class_action_domains[a1_class]
            a2_act = self.agent_class_action_domains[a2_class]
            joint_act_dom: ActionDomain = DiscreteActionDomain.join(a1_act, a2_act)
            assert joint_act_dom.discrete, 'CoordinationGraphSystem supports Discrete Action Domains only.'
            self._cached_joint_domains[joint_class] = joint_obs_dom, joint_act_dom
        return joint_act_dom, joint_obs_dom

    def observe_request(self):
        # We also want the joint features.
        return self._observation_sets

    # @profile
    def _get_edge_observation(self, observations, policy_group):
        """
        Combine single-agent & joint observations from observation map to make edge observations
        :param observations: single+joint observation map
        :param policy_group: policy group (edge pg) over which to get observation
        :return:
        """
        # A nasty operation :(
        # It is also dependent on DiscreteActionFeatureDomain being used!
        # And that the joint observation's action is the joint action.
        a1, a2 = policy_group.agents
        ac1, ac2 = self.agent_id_class_map[a1], self.agent_id_class_map[a2]
        od1, od2 = self.agent_class_observation_domains[ac1], self.agent_class_observation_domains[ac2]
        od12 = self.joint_agent_class_observation_domains[(ac1, ac2)]
        obs_1 = observations[a1].reshape(od1.packed_shape)
        obs_2 = observations[a2].reshape(od2.packed_shape)
        obs_joint = observations[(a1, a2)].reshape(od12.packed_shape)
        full_observation = DiscreteActionFeatureDomain.broadcast_joint(obs_1, obs_2) + (obs_joint,)
        edge_observation = np.hstack(full_observation).flatten()
        return edge_observation

    def translate_pg_signal(self,
                            a_observations: Dict[Union[Tuple[int, int], int], np.ndarray],
                            a_actions: Dict[int, np.ndarray],
                            a_rewards: Dict[Union[Tuple[int, int], int], Union[int, np.ndarray]]):
        # Called AFTER get_actions.
        # In CG-System, the PG ID is the Edge ID.
        # (note?) avoid coordinating agents with different objectives; CG-sys naively adds rewards along edges
        pg_observations: Dict[int, np.ndarray] = {}
        pg_rewards: Dict[int, Union[int, float]] = {}
        pg_actions: Dict[int, np.ndarray] = {}
        for policy_group in self.graph.edge_pg_map.values():
            pg_id = policy_group.pg_id
            a1, a2 = policy_group.agents
            if pg_id in self._last_pg_observations:
                pg_observations[pg_id] = self._last_pg_observations[pg_id]
            else:
                pg_observations[pg_id] = self._get_edge_observation(a_observations, policy_group)
            act_dom: DiscreteActionDomain = policy_group.policy.domain_act
            pg_actions[pg_id] = act_dom.make_compound_action(a_actions[a1], a_actions[a2])
            pg_rewards[pg_id] = a_rewards[a1] + a_rewards[a2]
        if self.use_nodes:
            for vertex_id, policy_group in self.graph.vertex_pg_map.items():
                pg_id = policy_group.pg_id
                a1 = policy_group.agents[0]
                pg_observations[pg_id] = a_observations[a1]
                pg_actions[pg_id] = a_actions[a1]
                pg_rewards[pg_id] = a_rewards[a1]
        return pg_observations, pg_actions, pg_rewards

    # @profile
    def get_actions(self, observations: Dict[Union[Tuple[int, int], int], np.ndarray], use_max: bool = False)\
            -> Dict[int, np.ndarray]:
        # Concat obs in the same way as we made the policy domains for our pseudo meta-agents (edges)
        # Dict from edge to array of shape [ar1, ar2] (Q). from [ar1*ar2].reshape(ar1, ar2...) (Q) for as-is.

        # TODO parallel evaluation of all policy groups
        all_action_q = {}  # This one has to be over edges.
        self._last_pg_observations.clear()
        for edge_id, policy_group in self.graph.edge_pg_map.items():
            edge_observation = self._get_edge_observation(observations, policy_group)
            pg_id = policy_group.pg_id
            self._last_pg_observations[pg_id] = edge_observation
            joint_act_dom: DiscreteActionDomain = policy_group.policy.domain_act
            all_action_q[edge_id] = policy_group.policy.eval(edge_observation).reshape(joint_act_dom.ranges)

        final_amap = {}
        if self.use_nodes:
            # Note: Dim 1 is a1, Dim 2 is a2 of edge. Output is along agent, so if it is a1, it should be broadcast
            #       to accumulate against dim 1.
            for agent_id, policy_group in self.graph.vertex_pg_map.items():
                edge_info = self.graph.agent_edges_info[agent_id]
                if len(edge_info) > 0:
                    q = policy_group.policy.eval(observations[agent_id])
                    split_node_q = q / self.graph.agent_degree[agent_id]
                    for edge_id, neighbor, primacy in edge_info:
                        all_action_q[edge_id] += split_node_q.T if primacy else split_node_q
                else:
                    # Dangling agents # TODO for now don't explore because the rest isn't exploring either.
                    final_amap[agent_id] = policy_group.policy.get_actions(observations[agent_id], use_max)

        # Function selected during init.
        action_map = self.action_selection(all_action_q)
        # Dangling agents if no node available
        if not self.use_nodes:
            for agent_id in self.agent_ids:
                if len(self.graph.agent_edges_info[agent_id]) == 0:
                    action_map[agent_id] = np.random.randint(0, self._agent_id_action_size_map[agent_id])
        # TODO (future, algorithmic) Figure out to expressively formulate CG-REINFORCE, CG-LSPI without needing to
        #      examine the class of the algorithm
        # (Make use of policy sampler within the action selection algorithms?)
        # if not use_max:
        #     for aid, v in action_map.items():
        #         if np.random.rand() > 0.9:
        #             action_map[aid] = np.random.randint(0, self._agent_id_action_size_map[aid])
        for agent_id, action in action_map.items():
            final_amap[agent_id] = np.array([action])
        return final_amap

    def action_selection(self, all_action_q: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Action selection technique for Coordination Graph Q Learning.
        :param all_action_q: Q-values for all joint actions at each edge.
        :return: Action map (scalar)
        """
        raise AttributeError("This function should be selected and assigned over in __init__.")

    def get_random_action_map(self):
        """
        Return a randomized action map.
        :return: Randomized action map
        """
        best_action_map = {}
        for agent_id, agent_class in self.agent_id_class_map.items():
            act_dom = self.agent_class_action_domains[agent_class]
            best_action_map[agent_id] = act_dom.random_sample()
        return best_action_map

    def _greedy_with_restarts(self, all_action_q: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Greedy action selection technique.

        __Not guaranteed to converge__ if random initialization enters an orbit around the true maximum.

        Consider a single agent pair. The q-values are in a square along the joint action space.
        Let us have some example q values 0, 1, 2.

        0000
        0101
        0020
        0000

        In this case, if we started with joint action (1,1) we will make a tied change to (1,3) and back again.

        We will never find (2,2) despite it being better, due to the limitation that we only make one change at a time.
        :param all_action_q:
        """
        best_action_map = self.get_random_action_map()
        next_action_map = copy.copy(best_action_map)  # ok to copy; we don't touch arrays, just refs in dict
        best_q, best_q_values = self.get_total_q(best_action_map, all_action_q)
        for i in range(0, self.num_iterations):
            # Tie-pick the [agent, action] pair which raises total q the most.
            tied_changes = []
            mut_best_q = best_q
            current_best_q = best_q
            current_best_q_values = copy.copy(best_q_values)
            for agent_id, agent_class in self.agent_id_class_map.items():
                # Find the action which raises q in the neighborhood the most
                sum_agent_proposals = np.zeros(self._agent_id_action_size_map[agent_id])
                agent_act_idx = np.r_[:self._agent_id_action_size_map[agent_id]]
                for edge_id in self.graph.agent_edges[agent_id]:
                    # all_action_q is nicely shaped so we can just sum the qs for all neighbour edges
                    edge = self.graph[edge_id]
                    if agent_id == edge[0]:
                        sum_agent_proposals += all_action_q[edge_id][:, best_action_map[edge[1]]]
                    else:
                        assert agent_id == edge[1]
                        sum_agent_proposals += all_action_q[edge_id][best_action_map[edge[0]], :]
                new_agent_q = sum_agent_proposals.max()
                mut_action = random.choice(agent_act_idx[sum_agent_proposals == new_agent_q])
                # Replace altered q-values with the sum of new q values over neighborhood.
                for edge_id in self.graph.agent_edges[agent_id]:
                    mut_best_q -= current_best_q_values[edge_id]
                mut_best_q += new_agent_q

                if mut_best_q == current_best_q:
                    tied_changes = [(agent_id, mut_action)]
                elif mut_best_q > current_best_q:
                    tied_changes.append((agent_id, mut_action))
                    current_best_q = mut_best_q

            next_action_map = self._greedy_q_try_mut(next_action_map, tied_changes)

            # Was overall improved? Set the best action map if it was.
            mut_best_q, mut_q_values = self.get_total_q(next_action_map, all_action_q)
            if mut_best_q > best_q:
                best_q = mut_best_q
                best_q_values = mut_q_values
                best_action_map = next_action_map
        return best_action_map

    def _greedy_q_try_mut(self, next_action_map, tied_changes):
        """
        Subroutine in greedy action selection.

        Examines tied_changes for proposals which differ from the current action map to mutate the action map,
            returning a random action map if none differ.
        :param next_action_map: Action map to mutate.
        :param tied_changes: List of proposed mutations.
        :return: Mutated or randomized action map.
        """
        # Do one better than stochastic by making SURE we're in a fixpoint by checking the rest if "same" was result
        # Try to mutate the action map from tied changes. Return random action map if tied changes indicate fixpoint.
        # "For... until... finally" sort of deal.
        np.random.shuffle(tied_changes)
        for mut_agent, mut_action in tied_changes:
            if mut_action != next_action_map[mut_agent]:
                next_action_map[mut_agent] = mut_action
                return next_action_map
        return self.get_random_action_map()

    def _max_plus(self, edge_act_q_dists: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Max-plus action selection algorithm.

        Centralized Max-Plus
        """

        best_action_map = {}
        best_overall_action_q = float('-inf')
        # If there's any good way to "compile" actions into indices, we can speed things up by using C-like pointers.
        # construct map : (agent, neighbor, action) -> int (0)
        messages = {a1: {a2: np.zeros(self._agent_id_action_size_map[a2])
                                for a2 in self.graph.agent_neighbors[a1]} for a1 in self.agent_ids}
        for _ in range(self.num_iterations):
            old_msg = copy.deepcopy(messages)
            for agent_id, edge_ids in self.graph.agent_edges.items():
                num_msg_vals = 0
                sum_msg_vals = 0
                neighbors = self.graph.agent_neighbors[agent_id]
                prev_msg_sum = sum([old_msg[a2][agent_id] for a2 in neighbors])
                for edge_id in edge_ids:
                    a1, a2 = self.graph[edge_id]
                    if a1 == agent_id:   # edge: (me, other)
                        neighbor = a2
                        edge_q_dist = edge_act_q_dists[edge_id]    # (me, other)
                    else:   # edge: (other, me)
                        neighbor = a1
                        edge_q_dist = edge_act_q_dists[edge_id].T  # (other, me) -> (me, other)
                    # Find q of best other action for each self action.
                    # (sum in centralized max-plus does not include msg_ji(a))
                    best_act_reward = (edge_q_dist - old_msg[neighbor][agent_id][None, :]).max(axis=0)
                    messages[agent_id][neighbor][:] = best_act_reward + prev_msg_sum
                    sum_msg_vals += messages[agent_id][neighbor].sum()
                    num_msg_vals += 1
                normalizing_constant = sum_msg_vals / num_msg_vals
                for neighbor in neighbors:
                    messages[agent_id][neighbor] -= normalizing_constant
            # Construct action map from the messages
            proposed_action_map = {}
            for agent_id in self.agent_ids:
                tied_actions = []
                best_reward = float('-inf')
                neighbors = self.graph.agent_neighbors[agent_id]
                for action in range(self._agent_id_action_size_map[agent_id]):
                    reward = sum([messages[neighbor][agent_id][action] for neighbor in neighbors])
                    if reward == best_reward:
                        tied_actions.append(action)
                    elif reward > best_reward:
                        tied_actions = [action]
                        best_reward = reward
                # Choose randomly from among the best actions
                proposed_action_map[agent_id] = random.choice(tied_actions) \
                    if len(tied_actions) == 1 else tied_actions[0]
            # Assign as best action map if better
            mut_action_q, mut_q_map = self.get_total_q(proposed_action_map, edge_act_q_dists)
            if mut_action_q > best_overall_action_q:
                best_overall_action_q = mut_action_q
                best_action_map = proposed_action_map
        return best_action_map

    def _agent_elimination(self, edge_act_q_dists: Dict[int, np.ndarray]):
        """
        Agent elimination action selection method. New and fresh for Python+Numpy. Super fast and polytime on treewidth.
        """
        return self.elimtree.agent_elimination(edge_act_q_dists)

    def get_edge_q(self, action_map, all_action_q, edge_id):
        """
        Get the q value for a single edge with the action map
        :param action_map: Action map to be evaluated
        :param all_action_q: Dictionary containing [ACT1, ACT2] q-value array for each edge
        :param edge_id: Edge id to evaluate
        :return: Edge q
        """
        a1, a2 = self.graph[edge_id]
        return all_action_q[edge_id][action_map[a1], action_map[a2]]

    def get_total_q(self, action_map, all_action_q):
        """
        Get the total q value over the entire graph for an action map.
        :param action_map: Action map to be evaluated
        :param all_action_q: Dictionary containing [JOINT_ACTION] q-value array for each edge
        :return: Total q and edge -> q dict map
        """
        q_map = {}
        q_total = 0
        for edge_id in self.graph.edge_ids:
            edge_q = self.get_edge_q(action_map, all_action_q, edge_id)
            q_map[edge_id] = edge_q
            q_total += edge_q
        return q_total, q_map

    def learn_update(self):
        # Just have to make sure the contents of the trajectories have already been fused correctly.
        # TODO (future, performance) parallel update; determine width of parallel update during init to set up here.
        for policy_group in self.graph.edge_pg_map.values():
            self.algorithm.update(policy_group.policy, policy_group.model, policy_group.trajectory)
        if self.use_nodes:
            for policy_group in self.graph.vertex_pg_map.values():
                self.algorithm.update(policy_group.policy, policy_group.model, policy_group.trajectory)

    def transfer_domain(self, domain_transfer_message: DomainTransferMessage) -> 'AgentSystem':
        assert domain_transfer_message.add_agent_class_id_map is None

        # TODO verify that this still works

        remap = domain_transfer_message.agent_remapping

        # Remap agents to new canon
        mapped_ids = domain_transfer_message.remap_id_list(self.agent_ids)
        # print('Remap: ' + str(remap))
        # print('Resulting mapped IDs: ' + str(mapped_ids))
        old_agent_id_class_map = copy.deepcopy(self.agent_id_class_map)
        old_action_size_map = copy.deepcopy(self._agent_id_action_size_map)

        # Change graph one agent at a time... order  does matter but we don't have methods to deal with deletion order
        # TODO (algorithmic? research?) deletion order selection
        # TODO (verify for sanity and correctness)
        del_order = reversed([(a_write, a_read) for a_write, a_read in remap.items()])
        for a_write, a_read in del_order:
            if a_read is None:
                # print('dropping agent {}'.format(a_write))
                self._drop_agent(a_write)
            # print('Resulting PG IDs: ' + ', '.join([str(pg.pg_id) for pg in self.policy_groups]))

        # mapped_ids can be used to access the old graph.
        remap_to_canon = {}
        new_agent_ids = []
        new_agent_id_class_map = {}
        new_agent_class_id_map = {}
        new_agent_action_size_map = {}
        for i, id_from in enumerate(mapped_ids):
            remap_to_canon[id_from] = i
            new_agent_ids.append(i)
            agent_class = old_agent_id_class_map[id_from]
            new_agent_id_class_map[i] = agent_class
            if agent_class not in new_agent_class_id_map:
                new_agent_class_id_map[agent_class] = []
            new_agent_class_id_map[agent_class].append(i)
            new_agent_action_size_map[i] = old_action_size_map[id_from]

        # Map new edges. Align these two lists.
        new_pgs = []
        new_edges = []
        new_edge_pgs = []
        for old_edge_id, old_edge in self.graph.labeled_edges:
            # Deleted edge should have already stopped existing via drop agent
            new_edge = [remap_to_canon[remap[old] if old in remap else old] for old in old_edge]
            new_edges.append(new_edge)
            pg = self.graph.edge_pg_map[old_edge_id]
            pg.agents = [remap_to_canon[remap[old] if old in remap else old] for old in pg.agents]
            new_pgs.append(pg)
            new_edge_pgs.append(pg)

        new_graph = CoordinationGraph(new_agent_id_class_map, new_edges)
        new_agent_pg_map = {}

        if self.use_nodes:
            for old, pg in self.graph.vertex_pg_map.items():
                new_id = remap_to_canon[remap[old] if old in remap else old]
                pg.agents = [new_id]
                new_pgs.append(pg)
                new_graph.assign_vertex_pg(new_id, pg)
                if new_id not in new_agent_pg_map:
                    new_agent_pg_map[new_id] = []
                new_agent_pg_map[new_id].append(pg)

        for new_edge_id, new_edge in new_graph.labeled_edges:
            pg = new_edge_pgs[new_edge_id]
            new_graph.assign_edge_pg(new_edge_id, pg)
            for agent in new_edge:
                if agent not in new_agent_pg_map:
                    new_agent_pg_map[agent] = []
                new_agent_pg_map[agent].append(pg)

        self.agent_ids = new_agent_ids
        self.agent_id_class_map = new_agent_id_class_map
        self.agent_class_id_map = new_agent_class_id_map
        self.policy_groups = new_pgs
        self.agent_policy_groups_map = new_agent_pg_map
        self.graph = new_graph
        self.elimtree = EliminationTree(self.graph)
        self._agent_id_action_size_map = new_agent_action_size_map
        self._observation_sets = [agent_id for agent_id in new_agent_id_class_map.keys()]
        self._observation_sets += [tuple(edge) for edge in new_graph.edges]

        return self

    def _drop_agent(self, agent_id: int):
        """
        Helper method for agent deletion. Applies the appropriate domain transfer function to update the system.
        :param agent_id:
        :return:
        """

        # Damaged neighbours
        widows = self.graph.agent_neighbors[agent_id]

        # TODO (future) dimensionality reduction for transfer...?

        # # # Constructive transfer function - no deletions
        if self.transfer_method == 'drop_removed' or self.transfer_method == 'drop_restart':
            # Well, that's that, for this one. We just drop the stuff. There may be dangling agents.......
            pass
        elif self.transfer_method == 'map_to_neighbors' or self.transfer_method == 'project_over_existing':
            added_to: Dict[Tuple[int, int], int] = {}   # Existing edges added to by neighbour clique
            adds: Dict[Tuple[int, int], List[Policy]] = {}       # Edges added to via algorithm
            for edge_id, other, primacy in self.graph.agent_edges_info[agent_id]:
                lost_policy: Policy = self.graph.edge_pg_map[edge_id].policy
                if self.transfer_method == 'map_to_neighbors':
                    # Map to neighbours
                    map_to = copy.copy(widows)
                    map_from = other
                else:
                    # Project to existing
                    map_to = copy.copy(self.graph.agent_neighbors[other])
                    map_from = agent_id
                map_to.remove(map_from)

                for mapped_agent in map_to:
                    clique_edge: Tuple[int, int] = tuple(sorted([mapped_agent, map_from]))
                    # Check if the affected edge already exists
                    # TODO (future) refactor to make cleaner
                    for ch_edge_id in self.graph.agent_edges[clique_edge[0]]:
                        if clique_edge[1] in self.graph[ch_edge_id]:
                            added_to[clique_edge] = ch_edge_id
                    if clique_edge not in adds: # Mark the edge (whether existent or not) for mapping to
                        adds[clique_edge] = [lost_policy]
                    else:   # Well, this shouldn't happen, but...
                        adds[clique_edge].append(lost_policy)

            # Average lost edges into clique on neighbours
            for edge_tup, policies in adds.items():
                # Add first basis
                if edge_tup not in added_to:
                    # IF NEW EDGE
                    new_edge_id = self._new_edge(edge_tup)
                    policy: Policy = self.graph.edge_pg_map[new_edge_id].policy
                    # TODO (warn) we should not map to a different AC-pair!
                    old_fa: ActionFreeLinearNpy = policies[0].function_approximator
                    fa: ActionFreeLinearNpy = policy.function_approximator
                    sum_weights = old_fa.get_weights()
                    policies = policies[1:]
                else:
                    # IF EXISTING EDGE
                    edge_id = added_to[edge_tup]
                    policy: Policy = self.graph.edge_pg_map[edge_id].policy
                    fa: ActionFreeLinearNpy = policy.function_approximator
                    sum_weights = fa.get_weights()
                # Add rest of bases and combine
                for prev_policy in policies:
                    old_fa: ActionFreeLinearNpy = prev_policy.function_approximator
                    sum_weights += old_fa.get_weights()
                # Should be, for "true map to neighbors" add the avg of remapped to the new.
                fa.set_weights(sum_weights / float(len(policies) + 1))

        # # # Perform deletion
        # Drop from observation sets
        for tup_int in self._observation_sets:
            if isinstance(tup_int, tuple) and agent_id in tup_int:
                self._observation_sets.remove(tup_int)
            elif isinstance(tup_int, int) and agent_id == tup_int:
                self._observation_sets.remove(tup_int)

        # Remove agent and associated edges, policygroup
        for deleted_pg in self.graph.remove_agent(agent_id):
            for member_agent in deleted_pg.agents:
                self.agent_policy_groups_map[member_agent].remove(deleted_pg)
            self.policy_groups.remove(deleted_pg)
        del self.agent_policy_groups_map[agent_id]
        del self._agent_id_action_size_map[agent_id]

        # If drop restart, re-initialize all PGs.
        if self.transfer_method == 'drop_restart':
            for pg in self.policy_groups:
                policy: Policy = pg.policy
                if isinstance(policy.function_approximator, AbstractPyTorchFA):
                     value_dict = policy.function_approximator.get_variable_vals()['model']
                else:
                     value_dict = policy.function_approximator.get_variable_vals()
               
                # Reset according to some rule. TODO generalize reset rule
                for k, v in value_dict.items():
                    v[:] = 0.1
                    value_dict[k] = v
                policy.function_approximator.set_variable_vals(value_dict)

        # Remove agent entirely
        agent_class = self.agent_id_class_map[agent_id]
        self.agent_class_id_map[agent_class].remove(agent_id)
        del self.agent_id_class_map[agent_id]
        self.agent_ids.remove(agent_id)

    def _new_edge(self, edge_tup: Tuple[int, int]) -> int:
        """
        Add a new edge to the system WITHOUT ADDING NEW AGENTS.
        Instantiates policy and algorithm appropriately.
        :param edge_tup: Tuple of new edge. Should be sorted (so edge_tup[0] < edge_tup[1]).
        :return: new_edge_id
        """
        # Find first unoccupied agent ID.
        new_e_id = self.graph.add_edge(edge_tup)
        policy_group = self._make_joint_policy_group(new_e_id, self.next_pg_id)
        self.policy_groups.append(policy_group)
        self.graph.assign_edge_pg(new_e_id, policy_group)
        for agent_id in policy_group.agents:
            self.agent_policy_groups_map[agent_id].append(policy_group)
        self.next_pg_id += 1
        return new_e_id
