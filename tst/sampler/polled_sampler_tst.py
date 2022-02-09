import unittest
from agentsystem import HierarchicalSystem
from domain.hierarchical_domain import action_hierarchy_from_config
import types
from config import Config
from controller.mdp_controller import MDPController
from controller.offline_controller import OfflineController
import json
import numpy as np
import os

class TestPolledSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        json_path = '../sample_configs/test_polled.json'

        cls.json_file = open(json_path, 'r')
        cls.data = json.load(cls.json_file)

        config = Config(OfflineController, cls.data, relative_path=json_path)
        cls.controller = OfflineController(config)
        cls.env: Environment = config.environment.module_class(config)
        cls.asys: HierarchicalSystem = cls.controller.asys
        cls.sampler = cls.controller.sampler

        cls.controller.run()

    @classmethod
    def setUp(self) -> None:
        #config = Config('..\sample_configs\debug_taxiworld.json')
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        cls.json_file.close()

    def test_inhibited_samples(self):
        """
        Ensures that all inhibited samples are in fact terminated
        :return:
        """
        error = False
        for inhibited_sample in self.controller.inhibited_samples:
            state = inhibited_sample[0]
            action = inhibited_sample[1]
            all_non_term = [self.controller.sampler.primitive_action_map[i] for i in self.controller.sampler.get_all_non_term_primitives(state, "Root", 0)]
            if action in all_non_term:
                error = True
            self.assertEqual(-20, inhibited_sample[2])

        self.assertFalse(error)


        for episode in self.controller.derived_samples_per_sample:
            count = 0
            action = -1
            for action_sample in episode:
                if action_sample[2] != -20:
                    action = action_sample[1]

                if action_sample[2] == -20:
                    count += 1

            if action in [0, 1, 2, 3]:
                self.assertEqual(count, 24)
            elif action == 4:
                self.assertEqual(count, 4)
            elif action == 5:
                self.assertEqual(count, 1)
            else:
                print(action, action_sample, episode)
                self.assertTrue(False)

        self.assertFalse(error)


    def test_duplicates(self):
        """
        Checks that none of the derived samples are duplicated for a single sample set
        :return:
        """
        for episode in self.controller.derived_samples_per_sample:
            # print(episode)
            for action_sample in episode:
                count = 0
                state = action_sample[0]
                action = action_sample[1]
                for other_samples in episode:
                    if all(other_samples[0] == state) and other_samples[1] == action and other_samples[2] == action_sample[2]:
                        count += 1
                self.assertEqual(count, 1)

    def test_abstract_samples(self):
        """
        Checks the number of abstracted samples
        Checks that all of the abstracted samples are non-terminated actions
        Checks that none of the abstracted samples are repeated in a single sample set
        :return:
        """
        error = False
        for abstract_sample in self.controller.abstract_samples:
            state = abstract_sample[0]
            action = abstract_sample[1]
            all_non_term = [self.controller.sampler.primitive_action_map[i] for i in
                            self.controller.sampler.get_all_non_term_primitives(state, "Root", 0)]
            if action not in all_non_term:
                error = True

        for episode in self.controller.derived_samples_per_sample:
            # print(episode)
            for action_sample in episode:
                if action_sample[2] != -20:
                    count = 0
                    state = action_sample[0]
                    action = action_sample[1]
                    for other_samples in episode:
                        # if all(other_samples[0] == state):
                        #     print(other_samples, action_sample)

                        if all(other_samples[0] == state) and other_samples[1] == action and other_samples[2] == action_sample[2]:
                            count += 1

                    self.assertEqual(count, 1)

        for episode in self.controller.derived_samples_per_sample:
            count = 0
            action = -1
            for action_sample in episode:
                if action_sample[2] != -20:
                    count += 1
                    action = action_sample[1]

            if action in [0, 1, 2, 3]:
                self.assertEqual(count, 23)
            elif action == 4:
                self.assertEqual(count, 3)
            elif action == 5:
                self.assertEqual(count, 0)
            else:
                print(action, action_sample, episode)
                self.assertTrue(False)

        self.assertFalse(error)

    def test_inhibited_abstract_sample(self):
        """
        Checks the inhibited samples from abstract samples, ensures that the actions are terminated
        :return:
        """
        error = False
        for inhibited_sample in self.controller.sampler.inhibited_abstract_samples:
            state = inhibited_sample[0]
            action = inhibited_sample[1]
            all_non_term = [self.controller.sampler.primitive_action_map[i] for i in
                            self.controller.sampler.get_all_non_term_primitives(state, "Root", 0)]
            if action in all_non_term:
                error = True
            self.assertEqual(-20, inhibited_sample[2])

        self.assertFalse(error)

    def test_combined_samples(self):
        """
        Checks all of the abstract samples, generates all of the terminated actions in the abstracted state,
        and makes sure that there has been an abstracted sample for each one.

        This checks the inhibited from abstracted samples
        :return:
        """
        for episode in self.controller.derived_samples_per_sample:
            for action_sample in episode:

                # Check for abstract samples:
                # If its an abstract sample, check to see if there is an inhibited action for the abstracted state
                if action_sample[2] != -20:
                    state = action_sample[0]
                    all_non_term = [self.controller.sampler.primitive_action_map[i] for i in
                                    self.controller.sampler.get_all_non_term_primitives(state, "Root", 0)]

                    # Find all of the terminated actions
                    term_acts = [i for i in self.controller.sampler.primitive_action_map.values() if i not in all_non_term]

                    # Check to see if there is an inhibited action made for this abstracted sample
                    for term_act in term_acts:
                        found = False
                        for other_samples in episode:
                            if all(other_samples[0] == state) and other_samples[1] == term_act and other_samples[2] == -20:
                                found = True
                        self.assertTrue(found)

if __name__ == '__main__':
    unittest.main()