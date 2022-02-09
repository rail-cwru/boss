import json
import unittest
import os
from config import Config
from controller.mdp_controller import MDPController
from environment.Wargus import Wargus


class WargusTest(unittest.TestCase):

    def setUp(self):
        json_path = '../sample_configs/test_wargus.json'
        # with open('bitflip_hierachical.json') as f:
        self.json_file = open(json_path, 'r')
        self.data = json.load(self.json_file)
        config = Config(MDPController, self.data, relative_path=json_path)

        # edict = json.load(f)
        # environment = type('environment', (object,), edict)
        # config = type('config', (object,), {'environment': environment})
        self.controller = MDPController(config)

        self.env: Environment = config.environment.module_class(config)
        self.asys: HierarchicalSystem = self.controller.asys

        self.controller.env._state["location"][0] = 0
        self.controller.env._state["location"][1] = 0

    def test_observation(self):
        obs = self.controller.env.observe()
        self.assertListEqual(list(obs[0]), [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1])

    def test_update(self):
        actions = {}
        actions[0] = [4]
        self.controller.env.update(actions)

        obs = self.controller.env.observe()
        self.assertListEqual(list(obs[0]), [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        self.controller.env._state["location"][0] = 3
        self.controller.env._state["location"][1] = 3

        actions[0] = [4]
        self.controller.env.update(actions)

        obs = self.controller.env.observe()
        self.assertListEqual(list(obs[0]), [3, 3, 1, 0, 0, 0, 0, 1, 1, 1, 1])

        actions[0] = [6]
        self.controller.env.update(actions)
        obs = self.controller.env.observe()

        self.assertListEqual(list(obs[0]), [3, 3, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    def test_reward(self):
        self.controller.env._state["location"][0] = 2
        self.controller.env._state["location"][1] = 2

        actions = {}
        actions[0] = [4]
        rew = self.controller.env.update(actions)
        self.assertEqual(rew[0], -10)

        actions[0] = [0]
        rew = self.controller.env.update(actions)
        self.assertEqual(rew[0], -1)

        self.controller.env._state["meet_wood_requirement"] = 0
        self.controller.env._meta_state["wood"] = 2

        self.controller.env._state['resource'] = 2
        self.controller.env._state["location"][0] = 3
        self.controller.env._state["location"][1] = 3

        actions[0] = [6]
        rew = self.controller.env.update(actions)
        self.assertEqual(rew[0], 49)

        actions[0] = [6]
        rew = self.controller.env.update(actions)
        self.assertEqual(rew[0], -11)

        self.controller.env._meta_state["wood"] = 3
        self.controller.env._state['resource'] = 2
        actions[0] = [6]
        rew = self.controller.env.update(actions)
        self.assertEqual(rew[0], -11)

    def tearDown(self) -> None:
        self.json_file.close()


if __name__ == '__main__':
    unittest.main()