
from environment import Environment
from common.aux_env_info import AuxiliaryEnvInfo
from domain.hierarchical_domain import HierarchicalActionDomain, action_hierarchy_from_config
from domain.observation import ObservationDomain
from common.properties import Properties
from typing import Dict, List, Tuple, TYPE_CHECKING, Union
from common.domain_transfer import DomainTransferMessage


class HierarchicalEnvironment(Environment):
    def __init__(self, config: 'Config'):
        super(HierarchicalEnvironment, self).__init__(config)

        self.derived_hierarchy = None
        self.derived_hierarchical_action_domain = None
        self.derived_hierarchical_observation_domain = None
        self.hierarchical_action_domain = None
        self.hierarchical_observation_domain = None
        self.hierarchy = None
        self._flat = True

    def abstract_all_observation_domains(self):
        h_obs_domain = {}
        for action in self.hierarchy.actions:
            if not self.hierarchy.actions[action]['primitive'] or 'state_variables' in self.hierarchy.actions[action]:
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.hierarchy.actions[action]['state_variables'])

        return h_obs_domain

    def abstract_all_second_observation_domains(self):
        h_obs_domain = {}
        for action in self.derived_hierarchy.actions:
            if not self.derived_hierarchy.actions[action]['primitive'] or 'state_variables' in self.derived_hierarchy.actions[action]:
                h_obs_domain[action] = self.abstracted_observation_domain(
                    self.derived_hierarchy.actions[action]['state_variables'])

        return h_obs_domain

    def get_auxiliary_info(self) -> AuxiliaryEnvInfo:
        if not self._flat:
            assert self.hierarchy, "Cannot get aux info for non-hierarchical learning, joint action domains not supported"
            return AuxiliaryEnvInfo(joint_observation_domains=self.hierarchical_observation_domain,
                                    hierarchy=self.hierarchical_action_domain,
                                    derived_hierarchy=self.derived_hierarchical_action_domain,
                                    derived_observation_domain=self.derived_hierarchical_observation_domain)

    def create_second_hierarchy(self, derived_h=None):
        if derived_h is not None:
            second_action_hierarchy = action_hierarchy_from_config(derived_h)
            self.derived_hierarchy = self.load_hierarchy(second_action_hierarchy)
            self.derived_hierarchical_action_domain = HierarchicalActionDomain(self.derived_hierarchy.root, self.derived_hierarchy)
            self.derived_hierarchical_observation_domain = self.abstract_all_second_observation_domains()

    def load_hierarchy(self, action_hierarchy):
        raise NotImplementedError

    def abstracted_observation_domain(self, state_variables: set) -> ObservationDomain:
        raise NotImplementedError

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[int]:
        return [0]

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    def set_seed(self, seed: int):
        # TODO: Does this have an additional RV?
        pass

    def set_initial_seed(self, seed: int):
        # This env does not use randomness at all
        pass

    def get_seed_state(self):
        return []

    def set_seed_state(self, seed_state):
        pass

    def transfer_domain(self, message: DomainTransferMessage) -> DomainTransferMessage:
        pass

    def visualize(self):
        raise NotImplementedError
