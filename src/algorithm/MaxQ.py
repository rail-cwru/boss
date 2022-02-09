from typing import List

import numpy as np

from domain.hierarchical_domain import HierarchicalActionDomain
from config import Config
#from algorithm import Algorithm
from algorithm.TemporalDifference import TemporalDifference
from common.trajectory import Trajectory
from model import Model
from policy import Policy
from policy import TabularPolicy
import math
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA

class MaxQ(TemporalDifference):
    """
    @author Eric Miller
    @contact edm54

    A hierarchical learning algorithm which has been (regrettably) embedded in agent system to some degree
    Based on the MaxQ paper learning algorithm
    Should be used with Hierarchical Agent Systems

    """

    def _get_action_value(self, policy: Policy, next_state: np.ndarray, next_action: np.ndarray):
        pass

    # Update the value (reward fn) of a primitive action using the MaxQQ Algo on page 3 of maxQ
    def primitive_update(self, policy: Policy, model: Model, trajectory: Trajectory):

        if isinstance(policy, TabularPolicy):
            # Update previous state
            state = np.reshape(trajectory.observations[-1], (1, -1))
            reward = trajectory.rewards[-1]
            current_value = policy.eval(state)

            # new val = (1-alpha)(cur val) + alpha * reward
            targets = self.learning_rate * (reward - current_value)
            policy.update_value_fn(state, targets)
        else:
            # TODO: work with other polcies/value fn
            raise NotImplementedError('Only supports tabular policy at the moment')

    def completion_function_update(self, completion_fn, pseudo_cf, v_s_prime: int, cf_value: int,
                                   pseudo_cf_value: int, n: int, child_observation):
        state = np.reshape(child_observation, (1, -1))
        cf_values = completion_fn.policy.eval(state).flatten()
        pseudo_cf_values = pseudo_cf.policy.eval(state).flatten()

        pseudo_reward = pseudo_cf.trajectory.rewards[-1]
        action = completion_fn.trajectory.actions[-1]
        discounted_factor = math.pow(self.discount_factor, n)

        cf_target = self.learning_rate * ((discounted_factor * (cf_value + v_s_prime)) - cf_values[action])
        pseudo_cf_target = (self.learning_rate * ((discounted_factor * (pseudo_reward + pseudo_cf_value + v_s_prime))
                            - pseudo_cf_values[action]))

        if isinstance(completion_fn.policy, TabularPolicy):

            completion_fn.policy.h_update(state, action, cf_target)
            pseudo_cf.policy.h_update(state, action, pseudo_cf_target)
        else:
            raise NotImplementedError("Only tabular supported as of now")
