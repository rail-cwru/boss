import unittest

from controller.mdp_controller import MDPController
from config import Config


class TestTemporalDifference(unittest.TestCase):
    def test_reinforce(self):
        config = Config('../../experiment_configs/test/REINFORCE/test.json')

        config.agentsystem.selection_method = 'greedy_with_restarts'
        controller = MDPController(config)
        observation_request = controller.asys.observe_request()
        agents_observation = controller.env.observe(observation_request)
        agent_action_map = controller.asys.get_actions(agents_observation)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
