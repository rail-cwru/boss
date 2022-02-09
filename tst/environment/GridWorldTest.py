import json
import unittest

from environment import GridWorld
import numpy as np

class GridWorldTest(unittest.TestCase):
    def setUp(self):
        with open('../../experiment_configs/environment/gridworld_test.json') as f:
            edict = json.load(f)
            environment = type('environment', (object,), edict)
            config = type('config', (object,), {'environment': environment})
            self.gw = GridWorld(config)

    def test_update_terminate(self):
        actions = {}
        actions[0] = np.array([3]) #move one agent onto good square, the other onto bad.
        actions[1] = np.array([0])
        reward = self.gw.update(actions=actions)
        self.assertEqual(reward, {0: 10, 1: -10})
        self.assertTrue(self.gw.done)

    def test_update_wall(self):
        actions = {}
        actions[0] = np.array([1])  # move one agent onto wall, the other onto empty.
        actions[1] = np.array([1])
        reward = self.gw.update(actions=actions)
        self.assertEqual(reward, {0: -1, 1: -1})
        self.assertFalse(self.gw.done)

    def test_observe(self):
        observation = self.gw.observe()[0]
        surroundings = observation[self.gw.agent_class_observation_domains[0].index_for_name("surroundings")]
        goal_surroundings = np.array(
            [1,1,1,
             0,0,2,
             3,1,3]
        ) #surroundings for first agent.
        self.assertTrue(np.array_equal(surroundings,goal_surroundings))

        agent_position = observation[self.gw.agent_class_observation_domains[0].index_for_name("agent_position")]
        goal_position = np.array([0,2])
        self.assertTrue(np.array_equal(agent_position, goal_position))
