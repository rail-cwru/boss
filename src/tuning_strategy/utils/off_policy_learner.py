from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from config import Config


class OffPolicyLearner(object):
    def __init__(self, policy_group: 'PolicyGroup', alg_config: 'Config', policy_config: 'Config', global_config: 'Config'):
        # Generate algorithm
        self.algorithm = alg_config.module_class(alg_config)

        # Establish policy
        self.policy = policy_config.module_class(policy_group.policy.domain_obs, policy_group.policy.domain_act, policy_config)

        # Base compiling off algorithm
        self.algorithm.compile_policy(self.policy)

        # Establish model if required
        self.model = alg_config.module_class.make_model(self.policy, global_config)

    def get_action_prob(self, state):
        return self.policy.get_action_probs(state)

    def learn(self, trajectory):
        self.algorithm.update(self.policy, self.model, trajectory)
