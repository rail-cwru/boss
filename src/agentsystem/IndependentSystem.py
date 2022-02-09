"""
Simple, independent AgentSystems that can support single agent or are naive for multiagent.
"""

from typing import Dict, List, Union, Tuple, Any

import numpy as np

from common.trajectory import Trajectory
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc
from . import AgentSystem
from .util import PolicyGroup
from common.domain_transfer import DomainTransferMessage
from common.properties import Properties
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from config import Config
from algorithm import Algorithm
import pickle


class IndependentSystem(AgentSystem, AbstractModuleFrame):
    """
    An Agent System where every single agent has its own independently learned policy.

    Pros: No headaches for scalability

    Cons: No coordination whatsoever - extremely large footprint for both time and space. Best only for single agent.
    """

    def __init__(self,
                 agent_id_class_map: Dict[int, int],
                 agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain],
                 auxiliary_data: Dict[str, Any],
                 config: Config):

        self.primitive_means = []
        self.putdown_means = []
        self.pickup_means = []

        super(IndependentSystem, self).__init__(agent_id_class_map, agent_class_action_domains,
                                                 agent_class_observation_domains, auxiliary_data, config)

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return []

    @classmethod
    def properties(cls) -> Properties:
        return Properties()

    def _make_policy_groups(self,
                            agent_id_class_map: Dict[int, int],
                            agent_class_action_domains: Dict[int, ActionDomain],
                            agent_class_observation_domains: Dict[int, ObservationDomain]) -> List[PolicyGroup]:
        self.means = []
        self.epi_means = []
        self.i = 0

        # Each of the unique agents has a unique policy group.
        # Agents in the same agent class should have same-shaped domains
        policy_groups = []
        for agent_id in agent_id_class_map.keys():
            # We know that each policy group only has one agent...
            policy_group_id = agent_id
            agent_class = self.agent_id_class_map[agent_id]

            # Single agents...
            obs_domain = self.agent_class_observation_domains[agent_class]
            act_domain = self.agent_class_action_domains[agent_class]
            policy = self.policy_class(obs_domain, act_domain, self.top_config)
            model = self.algorithm_class.make_model(policy, self.top_config)
            policy_groups.append(PolicyGroup(policy_group_id, [agent_id], policy, model, self.max_time))
        return policy_groups

    def get_actions(self, observations: Dict[int, np.ndarray], use_max: bool = False) -> Dict[int, np.ndarray]:
        # self.epi_means.append(np.mean(self.policy_groups[0].policy.table))

        # if not use_max:
        #     self.append_means()

        # TODO - eventual - accelerate data passing with cython structs?
        # Actions according to agent class.
        all_actions = {}
        for agent_id, observation in observations.items():
            # TODO refactor once observation-wrangling moved to PG
            pg = self.agent_policy_groups_map[agent_id][0]
            action = pg.policy.get_actions(observation, use_max)
            all_actions[agent_id] = action
        return all_actions

    def translate_pg_signal(self,
                            a_observations: Dict[int, np.ndarray],
                            a_actions: Dict[int, np.ndarray],
                            a_rewards: Dict[int, Union[int, np.ndarray]]):
        return a_observations, a_actions, a_rewards

    def learn_update(self):
        # Each... and every.... single... policy...... must be updated...
        # Luckily, it's that they are updated separately
        # TODO Parallelize
        for policy_group in self.policy_groups:
            self.algorithm.update(policy_group.policy, policy_group.model, policy_group.trajectory)

    def transfer_domain(self, domain_transfer_message: DomainTransferMessage) -> AgentSystem:
        # Only supports deletion (thus far)
        # never ever ever should there be a real runtime error due to bad config
        assert domain_transfer_message.add_agent_class_id_map is None

        # TODO consider refactoring common code in future?
        mapped_ids = domain_transfer_message.remap_id_list(self.agent_ids)

        # All deletions performed. Only remaps left.
        # Remap IDs (again) according to rule [remapped_ids] -> range(len(remapped_ids))
        new_agent_ids = []
        new_agent_id_class_map = {}
        new_agent_class_id_map = {}
        new_agent_policy_groups_map = {}
        new_policy_groups = []
        for canonical_id, id_from in enumerate(mapped_ids):
            new_agent_ids.append(canonical_id)
            agent_class = self.agent_id_class_map[id_from]
            new_agent_id_class_map[canonical_id] = agent_class
            if agent_class not in new_agent_class_id_map:
                new_agent_class_id_map[agent_class] = []
            new_agent_class_id_map[agent_class].append(canonical_id)
            pg = self.agent_policy_groups_map[id_from][0]
            new_policy_groups.append(pg)
            pg.agents = [canonical_id]
            new_agent_policy_groups_map[canonical_id] = [pg]

        self.agent_ids = new_agent_ids
        self.agent_class_id_map = new_agent_class_id_map
        self.agent_id_class_map = new_agent_id_class_map
        self.agent_policy_groups_map = new_agent_policy_groups_map
        self.policy_groups = new_policy_groups

        # Adding a new agent should be implemented in the future, and is really bothersome due to ID bumping
        # Also, the issue with creating a new PolicyGroup / Policy Object.
        # We might want to just set it as new initialization (sharing is purview of SharedSystem, not IS.)

        return self

    def append_means(self):
        pg_dict = self.policy_groups[0]

        all_means = np.mean(pg_dict.policy.table, axis = tuple([0, 1, 2]))

        self.primitive_means.append(round(np.mean(all_means[0:4]), 3))
        self.putdown_means.append(round(all_means[5], 3))
        self.pickup_means.append(round(all_means[4], 3))

