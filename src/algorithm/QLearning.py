import numpy as np

from algorithm.TemporalDifference import TemporalDifference
from policy import Policy

class QLearning(TemporalDifference):
    def _get_action_value(self, policy: Policy, next_state: np.ndarray, next_action: np.ndarray):
        prediction = policy.eval(next_state).flatten()
        return np.amax(prediction)
