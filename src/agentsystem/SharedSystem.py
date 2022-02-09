"""
Simple, independent AgentSystems that can support single agent or are naive for multiagent.
"""

from typing import Dict, List, Tuple, Union, Any
import copy

import numpy as np

from common.trajectory import Trajectory
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc
from . import AgentSystem
from .util import PolicyGroup
from config import Config
from common.domain_transfer import DomainTransferMessage
from common.properties import Properties
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from policy import Policy
from algorithm import Algorithm


class SharedSystem(AgentSystem, AbstractModuleFrame):
    """
    An Agent System where every single agent in the same agent class has the same (shared) policy.

    Pros: No headaches for scalability. Less footprint than IndependentSystem

    Cons: Requires figuring out batching scheme for "simultaneous" policy updates
    """

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return []

    @classmethod
    def properties(cls) -> Properties:
        return Properties() # obviated for now
        # Online is ok now that models are tied to policy-groups.

    def __init__(self,
                 agent_id_class_map: Dict[int, int],
                 agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain],
                 auxiliary_data: Dict[str, Any],
                 config: Config):
        # Convenience data for SharedSystem specifically that goes directly agentclass -> policy
        self.agent_class_policy_map: Dict[int, Policy] = {}
        super(SharedSystem, self).__init__(agent_id_class_map,
                                           agent_class_action_domains,
                                           agent_class_observation_domains,
                                           auxiliary_data,
                                           config)

    def _make_policy_groups(self,
                            agent_id_class_map: Dict[int, int],
                            agent_class_action_domains: Dict[int, ActionDomain],
                            agent_class_observation_domains: Dict[int, ObservationDomain]) -> List[PolicyGroup]:
        # Each of the unique agents has a unique policy group. But agents sharing agentclass will share policy.
        # Maps agent_class to policy
        class_policy_map = {}
        policy_groups = []
        # Since we have a one-to-one mapping from agent to policy group
        for agent_id in agent_id_class_map.keys():
            agent_class = self.agent_id_class_map[agent_id]
            if agent_class not in class_policy_map:
                # If it is a new agent class, make a new policy
                obs_domain = self.agent_class_observation_domains[agent_class]
                act_domain = self.agent_class_action_domains[agent_class]
                policy = self.policy_class(obs_domain, act_domain, self.top_config)
                class_policy_map = policy
            # Policy groups still exist for each separate agent.
            policy = class_policy_map[agent_class]
            model = self.algorithm_class.make_model(policy, self.top_config)
            policy_groups.append(PolicyGroup(agent_id, [agent_id], policy, model, self.max_time))
        return policy_groups

    def get_actions(self, observations: Dict[int, np.ndarray], use_max: bool = False) -> Dict[int, np.ndarray]:
        # Isn't this the same as in IndependentSystem? Can we refactor somehow?
        # Or can all these be moved to the base class? Keep an eye out.
        # TODO PARALLELIZE
        # TODO - eventual - accelerate data passing with cython structs?
        # Actions according to agent class.
        all_actions = {}
        for agent_id, observation in observations.items():
            pg = self.agent_policy_groups_map[agent_id][0]
            action = pg.policy.get_actions(observation, use_max)
            all_actions[agent_id] = action
        return all_actions

    def translate_pg_signal(self,
                            a_observations: Dict[int, np.ndarray],
                            a_actions: Dict[int, np.ndarray],
                            a_rewards: Dict[int, Union[int, np.ndarray]],
                            ):
        return a_observations, a_actions, a_rewards

    def learn_update(self):
        # Policies are the same for each agent class.
        # But this means the algorithm should have some sort of simultaneous update handling methodology.

        # Dict to (model, trajectory) pairs
        for policy_group in self.policy_groups:
            # The policy pointer is shared anyway, so we're fine.
            # TODO looks too similar...? maybe refactor
            self.algorithm.update(policy_group.policy, policy_group.model, policy_group.trajectory)

    def transfer_domain(self, domain_transfer_message: DomainTransferMessage) -> AgentSystem:
        # Supports deletion of individual agents only.
        # Make sure that deleting too many agents doesn't cause the agent class to be destroyed

        mapped_ids = copy.deepcopy(self.agent_ids)
        remaps: Dict[int, PolicyGroup] = {}
        deletes: Dict[int, PolicyGroup] = {}
        # Write references so we don't overwrite the originals while doing the remap.
        remap = domain_transfer_message.agent_remapping
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

        # Add case
        next_agent_id = len(new_agent_ids)
        for ac_new, num in domain_transfer_message.add_agent_class_id_map.items():
            # (new agent class) - not supported currently. maybe just initialize new?
            if ac_new not in self.agent_classes:
                raise NotImplementedError('Addition of new agent classes under SharedSystem not supported.')
            # (existing agent class)
            for i in range(num):
                new_id = next_agent_id + i
                new_agent_ids.append(new_id)
                policy = self.agent_class_policy_map[ac_new]
                model = self.algorithm_class.make_model(policy, self.top_config)
                pg = PolicyGroup(new_id, [new_id], policy, model, self.max_time)
                self.policy_groups.append(pg)
                new_agent_policy_groups_map[new_id] = [pg]
                if ac_new not in new_agent_class_id_map:
                    new_agent_class_id_map[ac_new] = []
                new_agent_class_id_map[ac_new].append(new_id)
                new_agent_id_class_map[new_id] = ac_new
            next_agent_id += num

        self.agent_ids = new_agent_ids
        self.agent_class_id_map = new_agent_class_id_map
        self.agent_id_class_map = new_agent_id_class_map
        self.agent_policy_groups_map = new_agent_policy_groups_map
        self.policy_groups = new_policy_groups

        return self

