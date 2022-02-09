import unittest
from agentsystem import HierarchicalSystem
from domain.hierarchical_domain import action_hierarchy_from_config
import types
from config import Config
from controller.mdp_controller import MDPController
import json
import numpy as np

class TestMaxQ(unittest.TestCase):

    @classmethod
    def setUp(self) -> None:
        # config = Config('..\sample_configs\debug_taxiworld.json')

        json_path = '../../sample_configs/test_hsys.json'
        self.json_file = open(json_path, 'r')
        self.data = json.load(self.json_file)
        config = Config(MDPController, self.data, relative_path=json_path)
        controller = MDPController(config)
        self.env: Environment = config.environment.module_class(config)
        self.asys: HierarchicalSystem = controller.asys

    def tearDown(self) -> None:
        self.json_file.close()

    def test_prim_update(self):
        c_pg = self.asys.completion_function_pg[0]['East']
        c_pg.policy.update_value_fn(np.asarray([[0,0]]), np.asarray([[.05 * 1]]))
        self.assertAlmostEqual(c_pg.policy.eval(np.asarray([[0,0]]))[0][0], .051)
        self.assertAlmostEqual(c_pg.policy.eval(np.asarray([[1, 0]]))[0][0], .001)
        c_pg.policy.update_value_fn(np.asarray([[0, 0]]), np.asarray([[.05 * -1]]))

        p_pg = self.asys.pseudo_reward_cf_pg[0]['East']
        p_pg.policy.update_value_fn(np.asarray([[0, 0]]), np.asarray([[.05 * 1]]))
        self.assertAlmostEqual(p_pg.policy.eval(np.asarray([[0, 0]]))[0][0], .051)
        p_pg.policy.update_value_fn(np.asarray([[0, 0]]), np.asarray([[.05 * -1]]))

        p_pg = self.asys.pseudo_reward_cf_pg[0]['Navigate__0_0_']
        p_pg.policy.update_value_fn(np.asarray([[0, 0]]), np.asarray([[.05 * 1]]))
        self.assertAlmostEqual(p_pg.policy.eval(np.asarray([[0, 0]]))[0][0], .051)
        self.assertAlmostEqual(p_pg.policy.eval(np.asarray([[1, 0]]))[0][0], .001)
        p_pg.policy.update_value_fn(np.asarray([[0, 0]]), np.asarray([[.05 * -1]]))

        p_pg = self.asys.pseudo_reward_cf_pg[0]['Root']
        obs = np.asarray([[0, 0, 0, 0, 0, 0, -1]])
        p_pg.policy.update_value_fn(obs, np.asarray([[.05 * 10]]))
        self.assertAlmostEqual(p_pg.policy.eval(obs)[0][0], .501)

        obs = np.asarray([[0, 1, 0, 0, 0, 0, -1]])
        self.assertAlmostEqual(p_pg.policy.eval(obs)[0][0], .001)

    def test_cf_update(self):
        obs = np.asarray([[0, 0, 0, 0, 0, 0, -1]])
        p_pg = self.asys.pseudo_reward_cf_pg[0]['Get']
        c_pg = self.asys.completion_function_pg[0]['Get']
        v_s_prime = .1
        cf_value = .5
        pseudo_cf_value = .5
        n = 1

        self.asys.current_reward_dict[0]['Navigate__0_0_'] = 0
        self.asys.current_reward_dict[0]['Get'] = 10

        #self.asys.current_reward_dict[agent_id][action] = 0
        self.asys.current_action_dict[0]['Get'] = 0
        self.asys.abstracted_obs_dict[0]['Get'] = 0
        self.asys.child_terminated_dict[0]['Get'] = 'Navigate__0_0_'

        self.asys.prepare_child('Navigate__0_0_', 'Get', 0, np.asarray([0, 0]))

        self.asys.algorithm.completion_function_update(c_pg, p_pg, v_s_prime, cf_value, pseudo_cf_value, n, obs)
        self.assertAlmostEqual(p_pg.policy.eval(obs)[0][0], 5.15075)
        self.assertAlmostEqual(c_pg.policy.eval(obs)[0][0], .15075)

if __name__ == '__main__':
    unittest.main()
