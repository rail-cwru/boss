import numpy as np
from config import Config, AbstractModuleFrame, ConfigItemDesc
from agentsystem.HierarchicalSystem import HierarchicalSystem

class BaseHierarchicalSampler(HierarchicalSystem, AbstractModuleFrame):
    """
    An agent system designed for random hierarchical policy sampling for offline learning
    """

    def _create_state_visits(self):
        """
        Initialize state visits arrays for both the general policy and for each subtask including primitives
        """

        self.state_action_basis = ExactBasis(np.asarray(self.num_states), self.num_actions)
        self.state_action_visits = np.zeros(self.state_action_basis.size(), np.int64)

        self.state_action_basis_map = {}
        self.state_action_visits_map = {}

        for name, pg in self.completion_function_pg[0].items():
            # if not self.is_primitive(name) or self.use_primitive_distribution:
            if self.collect_abstract:
                # Use abstracted state space
                obs_domain = pg.policy.domain_obs
                num_states = [domain_item.num_values() for domain_item in
                              obs_domain.items]
            else:
                # Use entire state since abstraction not being used
                num_states = self.num_states

            num_actions = pg.policy.domain_act.full_range
            self.state_action_basis_map[name] = ExactBasis(np.asarray(num_states), num_actions)
            self.state_action_visits_map[name] = np.zeros(self.state_action_basis_map[name].size(), np.int64)

    def get_derived_samples(self, last_obs, sucessful_traj):
        """
        Returns the derived samples
        :return:
        """
        if not sucessful_traj:
            self.add_s_prime(last_obs[0], 0)
        else:
            self.add_s_prime(None, 0)
        return self.inhibited_samples[0], self.abstracted_samples[0]