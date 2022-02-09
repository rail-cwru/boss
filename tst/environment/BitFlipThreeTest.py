import json
import unittest
import os
from config import Config
from controller.mdp_controller import MDPController
from environment.BitFlip import BitFlip

class BitFlipTest(unittest.TestCase):
    def setUp(self):
        json_path = '../sample_configs/test_bitflip_three.json'
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

        # Don't know why the previously called method for setting the state is not working...
        self.controller.env.state_val = [1, 0, 0, 1, 1, 0, 1, 0]

    def tearDown(self) -> None:
        self.json_file.close()

    def test_observe(self):
        obs = self.controller.env.observe()
        self.assertListEqual(list(obs[0][0:8]), [1, 0, 0, 1, 1, 0, 1, 0])
        self.assertListEqual(list(obs[0][8:16]), [1, 0, 0, 1, 1, 0, 1, 0])
        self.assertListEqual(list(obs[0][16:24]), [0, 1 ,1, 1, 1, 1, 1, 1])

        self.controller.env.state_val = [0, 0, 0, 0, 1, 0, 1, 0]
        obs = self.controller.env.observe()
        self.assertListEqual(list(obs[0][0:8]), [0, 0, 0, 0, 1, 0, 1, 0])
        self.assertListEqual(list(obs[0][8:16]), [0, 0, 0, 0, 1, 0, 1, 0])
        self.assertListEqual(list(obs[0][16:24]), [0, 0, 0, 0, 0, 1, 1, 1])


    def test_update(self):

        self.controller.env.state_val = [0, 0, 0, 0, 0, 0, 0, 0]
        actions = {}
        actions[0] = [3]
        self.controller.env.update(actions)

        # Test a bitflip
        self.assertListEqual(self.controller.env.state_val,  [0, 0, 1, 1, 1, 0, 0, 0])

        actions[0] = [1]
        self.controller.env.update(actions)
        self.assertListEqual(self.controller.env.state_val, [1, 1, 0, 1, 1, 0, 0, 0])

        actions[0] = [0]
        self.controller.env.update(actions)
        self.assertListEqual(self.controller.env.state_val, [0, 0, 0, 1, 1, 0, 0, 0])

        actions[0] = [4]
        self.controller.env.update(actions)
        self.assertListEqual(self.controller.env.state_val, [0, 0, 0, 0, 0, 1, 0, 0])

        actions[0] = [6]
        self.controller.env.update(actions)
        self.assertListEqual(self.controller.env.state_val, [0, 0, 0, 0, 0, 0, 1, 1])

        actions[0] = [7]
        self.controller.env.update(actions)
        self.assertListEqual(self.controller.env.state_val, [0, 0, 0, 0, 0, 0, 0, 0])


        # # Test a bad flip
        # actions[0] = [4]
        # self.controller.env.update(actions)
        # self.assertListEqual(self.controller.env.state_val, [1, 1, 1, 1, 1, 0, 1, 0])
        #
        # # Test a second bad flip
        # actions[0] = [7]
        # self.controller.env.update(actions)
        # self.assertListEqual(self.controller.env.state_val, [1, 1, 1, 1, 1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()