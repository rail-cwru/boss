# Verify the action selection techniques.

from typing import Dict, Any
from controller.mdp_controller import MDPController
from config import Config
import numpy as np


def test_driftauto():
    config = Config('../../experiment_configs/debug_drift.json')

    controller = MDPController(config)

    best_weights = np.array([[-.1, -1], [.1, 1]])
    weights = -best_weights

    # OPTIMAL POLICY
    # def wrapper(obs: Dict[Any, np.ndarray], act: Dict[Any, np.ndarray]):
    #     drift = np.dot([0.1, 1.0], obs[0])
    #     return {0: [(drift < 0) * 1]}
    # controller.on_action = wrapper

    ## OPTIMAL WEIGHTS
    def fix_weights():
        controller.asys.policy_groups[0].policy.function_approximator.set_weights(weights)

    def intercept_update(obs):
        fix_weights()
        return obs

    # controller.on_observe = intercept_update
    controller.before_run = fix_weights

    controller.run()


if __name__ == '__main__':
    test_driftauto()
