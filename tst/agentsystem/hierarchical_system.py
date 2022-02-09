import unittest
from agentsystem import HierarchicalSystem
from domain.hierarchical_domain import action_hierarchy_from_config
import types
from config import Config
from controller.mdp_controller import MDPController
import json
import numpy as np

class TestHierarchicalSystem(unittest.TestCase):

    @classmethod
    def setUp(self) -> None:
        #config = Config('..\sample_configs\debug_taxiworld.json')


        json_path = '../../sample_configs/test_hsys.json'
        self.json_file = open(json_path, 'r')
        self.data = json.load(self.json_file)
        config = Config(MDPController, self.data, relative_path=json_path)
        controller = MDPController(config)
        self.env: Environment = config.environment.module_class(config)
        self.asys: HierarchicalSystem = controller.asys

    def tearDown(self) -> None:
        self.json_file.close()

    def test_term(self):
        self.asys.action_stack_dict[0] = ['Root', 'Get', 'Navigate__0_0_', 'East']
        # self.asys.current_reward_dict[0]['East'] = -1
        # self.asys.current_reward_dict[0]['Navigate__0_0_'] = 0
        obs = [0, 0, 0, 0, 2, 2, -1]
        self.assertTrue(self.asys.is_terminated('East', obs, 0))
        self.assertTrue(self.asys.is_terminated('Navigate__0_0_', obs, 0))
        self.assertFalse(self.asys.is_terminated('Root', obs, 0))
        self.assertFalse(self.asys.is_terminated('Get', obs, 0))

        obs = [1, 0, 0, 0, 2, 2, -1]
        self.assertFalse(self.asys.is_terminated('Navigate__0_0_', obs, 0))

        obs = [0, 0, 0, 0, 2, 2, -1]
        self.assertFalse(self.asys.is_terminated('Get', obs, 0))
        self.assertTrue(self.asys.is_terminated('Put', obs, 0))

        obs = [2, 2, 2, 2, 0, 0, 0]
        self.assertTrue(self.asys.is_terminated('Get', obs, 0))
        self.assertFalse(self.asys.is_terminated('Root', obs, 0))
        self.assertFalse(self.asys.is_terminated('Put', obs, 0))

        obs = [2, 2, 0, 0, 2, 2, -2]
        self.assertTrue(self.asys.is_terminated('Put', obs, 0))
        self.assertTrue(self.asys.is_terminated('Root', obs, 0))

    def test_non_term_action(self):
        obs = [2, 2, 0, 0, 2, 2, -1]
        self.assertEqual(self.asys.get_non_term_action(obs, 'Root', 0)[0], 'Get')
        self.assertNotEqual(self.asys.get_non_term_action(obs, 'Get', 0)[0], 'Navigate__2_2_')

        obs = [2, 2, 0, 0, 2, 2, 0]
        self.assertEqual(self.asys.get_non_term_action(obs, 'Root', 0)[0], 'Put')


    def test_sample_actions(self):
        obs = [0, 0, 0, 0, 0, 0, -1]
        self.assertAlmostEqual(self.asys.sample_actions(obs, 'Root', 0, use_max=True)[1], .004)
        self.assertAlmostEqual(self.asys.sample_actions(obs, 'Get', 0, use_max=True)[1], .003)
        self.assertAlmostEqual(self.asys.sample_actions(obs, 'Get', 0, use_max=True, use_pseudo=True)[1], .003)

        self.asys.completion_function_pg[0]['East'].policy.update_value_fn(np.asarray([[0,0]]), np.asarray([[.05 * 1]]))
        self.assertAlmostEqual(self.asys.sample_actions(obs, 'Get', 0, use_max=True)[1], .053)
        self.asys.completion_function_pg[0]['East'].policy.update_value_fn(np.asarray([[0, 0]]),
                                                                           np.asarray([[.05 * -1]]))
        
        self.asys.completion_function_pg[0]['Navigate__0_0_'].policy.h_update(np.asarray([[0,0]]), np.asarray([0]), np.asarray([[.05 * 10]]))
        self.assertAlmostEqual(self.asys.sample_actions(obs, 'Get', 0, use_max=True)[1], .503)
        self.assertAlmostEqual(self.asys.sample_actions(obs, 'Navigate__0_0_', 0, use_max=True)[1], .002)

        self.asys.pseudo_reward_cf_pg[0]['Navigate__0_0_'].policy.h_update(np.asarray([[0, 0]]), np.asarray([0]),
                                                                              np.asarray([[.05 * 10]]))
        self.assertAlmostEqual(self.asys.sample_actions(obs, 'Get', 0, use_max=True)[1], .503)
        self.assertAlmostEqual(self.asys.sample_actions(obs, 'Navigate__0_0_', 0, use_max=True)[1], .502)
        self.assertEqual(self.asys.get_action_children('Get')[self.asys.sample_actions(obs, 'Get', 0, use_max=True)[0][0]], 'Navigate__0_0_')
        self.assertEqual(self.asys.sample_actions(obs, 'Navigate__0_0_', 0, use_max=True)[0], [0])

        self.asys.completion_function_pg[0]['Navigate__0_0_'].policy.h_update(np.asarray([[0, 0]]), np.asarray([0]),
                                                                              np.asarray([[.05 * -10]]))

    def test_eval_child(self):
        obs = np.asarray([0, 0, 0, 0, 0, 0, -1])
        e_root = self.asys.eval_children(obs, 'Root', 0)
        self.assertEqual(len(e_root), 2)
        self.assertAlmostEqual(e_root[0], e_root[1])
        self.assertAlmostEqual(e_root[0], .004)

        e_get = self.asys.eval_children(obs, 'Get', 0)
        self.assertEqual(len(e_get), 3)
        self.assertAlmostEqual(min(e_get), .002)

    def test_eval_max_node(self):
        obs = np.asarray([0, 0, 0, 0, 0, 0, -1])
        e_root = self.asys.eval_max_node(obs, 'Root', 0)
        self.assertAlmostEqual(e_root, .004)

        e_east = self.asys.eval_max_node(obs, 'East', 0)
        self.assertAlmostEqual(e_east, .001)
        self.asys.completion_function_pg[0]['East'].policy.update_value_fn(np.asarray([[0, 0]]),
                                                                           np.asarray([[.05 * 1]]))
        e_east = self.asys.eval_max_node(obs, 'East', 0)
        self.assertAlmostEqual(e_east, .051)

        e_nav = self.asys.eval_max_node(obs, 'Navigate__0_0_', 0)
        self.assertAlmostEqual(e_nav, .052)

        e_get = self.asys.eval_max_node(obs, 'Get', 0)
        self.assertAlmostEqual(e_get, .053)

        self.asys.completion_function_pg[0]['East'].policy.update_value_fn(np.asarray([[0, 0]]),
                                                                           np.asarray([[.05 * -1]]))
    def test_is_primitive(self):
        self.assertTrue(self.asys.is_primitive('Pickup'))
        self.assertTrue(self.asys.is_primitive('Putdown'))
        self.assertTrue(self.asys.is_primitive('South'))
        self.assertTrue(self.asys.is_primitive('East'))
        self.assertTrue(self.asys.is_primitive('West'))
        self.assertTrue(self.asys.is_primitive('North'))
        self.assertFalse(self.asys.is_primitive('Put'))
        self.assertFalse(self.asys.is_primitive('Get'))
        self.assertFalse(self.asys.is_primitive('Root'))
        self.assertFalse(self.asys.is_primitive('Navigate__0_0_'))
    
    def test_term_predicate(self):
        obs = np.asarray([0, 0, 0, 0, 0, 0, -1])
        self.asys.child_terminated_dict[0]['Get'] = 'Navigate__0_0_'
        self.asys.child_terminated_dict[0]['Navigate__0_0_'] = 'East'
        self.asys.child_terminated_dict[0]['Root'] = 'Get'

        rew = self.asys.check_pseudo_reward('Navigate__0_0_', obs, 0)
        self.assertEqual(rew, 20)

        rew = self.asys.check_pseudo_reward('East', obs, 0)
        self.assertEqual(rew, 0)

        rew = self.asys.check_pseudo_reward('Get', obs, 0)
        self.assertEqual(rew, 0)
        with self.assertRaises(ValueError) as context:
            self.asys.check_pseudo_reward('Root', obs, 0)

        self.assertTrue('Pseudo Reward only for terminated actions' in str(context.exception))

    def test_clr_term(self):
        self.asys.action_stack_dict[0] = ['Root', 'Get', 'Navigate__0_0_', 'East']
        self.asys.child_terminated_dict[0]['Get'] = 'Navigate__0_0_'
        self.asys.child_terminated_dict[0]['Navigate__0_0_'] = 'East'
        self.asys.child_terminated_dict[0]['Root'] = 'Get'
        agent_id = 0
        for action in self.asys.action_stack_dict[0]:
            self.asys.current_reward_dict[agent_id][action] = 0
            self.asys.current_action_dict[agent_id][action] = 0
            self.asys.abstracted_obs_dict[agent_id][action] = 0
            self.asys.action_sequence_dict[agent_id][action] = [0]

        self.asys.clear_terminated_actions(0)
        self.assertEqual(self.asys.action_stack_dict[0], ['Root'])

        self.asys.action_stack_dict[0] = ['Root', 'Get', 'Navigate__0_0_', 'East']
        self.asys.child_terminated_dict[0].pop('Root')

        agent_id = 0
        for action in self.asys.action_stack_dict[0]:
            self.asys.current_reward_dict[agent_id][action] = 0
            self.asys.current_action_dict[agent_id][action] = 0
            self.asys.abstracted_obs_dict[agent_id][action] = 0
            self.asys.action_sequence_dict[agent_id][action] = [0]

        self.asys.clear_terminated_actions(0)
        self.assertEqual(self.asys.action_stack_dict[0], ['Root', 'Get'])

        self.asys.action_stack_dict[0] = ['Root', 'Get', 'Navigate__0_0_', 'East']
        self.asys.child_terminated_dict[0].pop('Get')

        agent_id = 0
        for action in self.asys.action_stack_dict[0]:
            self.asys.current_reward_dict[agent_id][action] = 0
            self.asys.current_action_dict[agent_id][action] = 0
            self.asys.abstracted_obs_dict[agent_id][action] = 0
            self.asys.action_sequence_dict[agent_id][action] = [0]

        self.asys.clear_terminated_actions(0)
        self.assertEqual(self.asys.action_stack_dict[0], ['Root', 'Get','Navigate__0_0_'])

    def test_slicer(self):
        obs = np.asarray([1, 1, 0, 0, 2, 2, -1])
        pg = self.asys.completion_function_pg[0]['Navigate__0_0_']
        self.assertListEqual(list(self.asys.slice_observation(pg, obs, 0)), [1,1])

        pg = self.asys.completion_function_pg[0]['Get']
        self.assertListEqual(list(self.asys.slice_observation(pg, obs, 0)), [1, 1, 0, 0, -1])

        pg = self.asys.completion_function_pg[0]['Root']
        self.assertListEqual(list(self.asys.slice_observation(pg, obs, 0)), list(obs))

        pg = self.asys.completion_function_pg[0]['Put']
        self.assertListEqual(list(self.asys.slice_observation(pg, obs, 0)), [1, 1, 2, 2, -1])

    def test_child_parent(self):
        self.asys.action_stack_dict[0] = ['Root', 'Get', 'Navigate__0_0_', 'East']
        self.assertCountEqual(self.asys.get_action_children('Root'), ['Get', 'Put'])
        self.assertCountEqual(self.asys.get_action_children('Get'), ['Pickup', 'Navigate__0_0_', 'Navigate__2_2_'])

        self.assertEqual(self.asys.get_parent('Get', 0), 'Root')
        self.assertEqual(self.asys.get_parent('Navigate__0_0_', 0), 'Get')
        self.assertEqual(self.asys.get_parent('East', 0), 'Navigate__0_0_')

if __name__ == '__main__':
    unittest.main()