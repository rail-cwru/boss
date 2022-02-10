from typing import Dict, List, Tuple, Union, Any

import numpy as np
from domain.ActionDomain import ActionDomain
from domain.observation import ObservationDomain
from config import Config, AbstractModuleFrame, ConfigItemDesc
from common.predicates import predicates
from config import Config, checks
from policy.function_approximator.basis_function.ExactBasis import ExactBasis
from agentsystem.util import PolicyGroup
from sampler.PolledSampler import PolledSampler
import copy


def _valid_action_hierarchy(ah):
    # TODO
    return True

class FlattenedPolledSampler(PolledSampler):

    """
    This is the flattened sampler algorithm which runs BUF and TDF sampling.
    The flattened hierarchy is automatically generated by giving the algorithm either "BUF" or "TDF" and
        the number of iterations

    This hierarchy will be saved in the results folder.
    If keep_navigate is True, the navigate actions in the hierarchy (taxi and Heist) will be preserved.
        Can be helpful to avoid the movement actions from overpowering the other actions after flattening
            i.e one navigate action has 4 children
        This was experimented with but may not have a large impact.

    NOTE: If you make a new environment, you may need to define an additional variable in
         derived_make_additional_var
         For example, in txi, the target variable needs to be specially handled since it is not in the hierarchy and has
            a special meaning

    """

    def get_class_config() -> List[ConfigItemDesc]:
        return[
            ConfigItemDesc('collect_abstract',
                           checks.boolean,
                           'Collect abstract samples',
                           default=True,
                           optional=True),
            ConfigItemDesc('collect_inhibited',
                           checks.boolean,
                           'Collect inhibited samples',
                           default=True,
                           optional=True),
            ConfigItemDesc('min_reward',
                           checks.negative_integer,
                           'Minimum Reward for Inhibited Samples',
                           default=-200,
                           optional=True),
            ConfigItemDesc('use_primitive_dist',
                           checks.boolean,
                           'Use the number of times a primitive action has been sampled, regardless of parent',
                           default=False,
                           optional=True),
            ConfigItemDesc('optimistic', checks.boolean, 'Use balanced wandering', default=False, optional=True),
            ConfigItemDesc('save_target',
                           checks.boolean,
                           'Save distribution as target distro for kl divergence',
                           default=False,
                           optional=True),
            ConfigItemDesc('save_name', checks.string, 'Name for saving target distribution for future testing',
                           default='chair_target.list', optional=True),
            ConfigItemDesc('flattener', checks.string, 'Which flattener to use', optional=False),
            ConfigItemDesc("iterations", checks.positive_integer, 'How many times to flatten', default=1, optional=True),
            ConfigItemDesc("keep_navigate", checks.boolean, 'Keep navigate actions when flattening', default=False,
                           optional=True)
        ]

    def __init__(self, agent_id_class_map: Dict[int, int], agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain], auxiliary_data: Dict[str, Any],
                 config: Config):
        if config.sampler.name != "BOSS":
            assert config.sampler.flattener == "BUF" or config.sampler.flattener == "TDF", 'Flattener must be BUF or TDF'

        self.derived_samples_hierarchy = auxiliary_data.derived_hierarchy
        self.derived_observation_domain = auxiliary_data.derived_observation_domain

        super(FlattenedPolledSampler, self).__init__(agent_id_class_map, agent_class_action_domains,
                                                     agent_class_observation_domains, auxiliary_data, config)

        # if self.collect_abstract and self.collect_inhibited:
        self.derived_inhibited_actions_basis = ExactBasis(np.asarray(self.num_states), 1)
        self.derived_inhibited_actions_arr = [[] for _ in range(self.inhibited_actions_basis.size())]

    def build_derived_pg_map(self, current_node: str, agent_class_observation_domains: Dict[int, ObservationDomain],
                      agent_class_action_domains: Dict[int, ActionDomain], agent_id: int, agent_class: int, pg_dict={},
                      pseudo_pg_dict={}, primitive=None) -> Tuple[Dict[str, PolicyGroup], Dict[str, PolicyGroup]]:

        if not primitive or current_node in self.derived_observation_domain:
            obs_domain = self.derived_observation_domain[current_node]
        else:
            # Primitive actions get obs domain of parent
            obs_domain = self.derived_observation_domain[primitive]
        action_domain = self.derived_samples_hierarchy.domain_for_action(current_node)

        # Checks if policy already exists for sharing
        if current_node in self.derived_policy_map[agent_class]:
            policy = self.derived_policy_map[agent_class][current_node]
        else:
            policy = self.policy_class(obs_domain, action_domain, self.top_config)
            self.derived_policy_map[agent_class][current_node] = policy

        model = self.algorithm_class.make_model(policy, self.top_config)
        pg_dict[current_node] = PolicyGroup(current_node, [agent_id], policy, model, self.max_time)

        # Recurse on children
        for child in self.get_derived_action_children(current_node):
            if child not in pg_dict:
                if self.derived_is_primitive(child):
                    primitive = current_node
                else:
                    primitive = None
                pg_dict = self.build_derived_pg_map(child, agent_class_observation_domains,
                                                    agent_class_action_domains, agent_id, agent_class,
                                                    pg_dict=pg_dict, pseudo_pg_dict=pseudo_pg_dict,
                                                    primitive=primitive)
        return pg_dict

    def generate_inhibited_from_abstract(self, agent_id):
        for abstract_sample in self.current_abstracted_samples[agent_id]:
            state = abstract_sample[0]
            s_prime = abstract_sample[-1]

            if len(self.num_states) != len(state):
                mapped_state = self._map_table_indices(state[:], self.derived_policy_map[agent_id]["Root"].policy)
            else:
                mapped_state = state[:]

            inhibited_index = self.derived_inhibited_actions_basis.get_state_action_index(mapped_state, 0)

            if not self.derived_inhibited_actions_arr[inhibited_index]:
                non_term_prim = self.get_all_derived_non_term_primitives(state, "Root", agent_id)
                term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
                self.derived_inhibited_actions_arr[inhibited_index] = term_prim
            else:
                term_prim = self.derived_inhibited_actions_arr[inhibited_index]

            for inhibited_action in term_prim:
                prim_action = self.primitive_action_map[inhibited_action]
                inhibited_samples = [mapped_state, np.asarray([prim_action]), self.min_reward, s_prime]
                self.current_inhibited_abstracted_samples[agent_id].append(inhibited_samples)

    def generate_abstract_samples(self, observation, policy_group, agent_id, primitive_action, current_node="Root"):
        """
        Generates abstract samples by changing the values of state variables that
        the hierarchy determines are irrelevant

        :param observation: Current (full) observation
        :param policy_group: current policy group
        :param agent_id: agent-id for policy group
        :param primitive_action: primitive action of sample
        :return:
        """
        root_pg = self.derived_policy_map[agent_id]['Root']
        state_list, s_prime_list = self.get_irrelevant_states(observation, policy_group, agent_id)
        for ind, state in enumerate(state_list):

            if len(self.num_states) != len(observation):
                mapped_state = self._map_table_indices(state, root_pg.policy)
            else:
                mapped_state = self.slice_observation(root_pg, state, agent_id)

            sample = [mapped_state, np.asarray([primitive_action]), None, s_prime_list[ind]]
            self.current_abstracted_samples[agent_id].append(sample)
            s_prime = s_prime_list[ind]

            if self.collect_inhibited and self.collect_abstract:
                inhibited_index = self.derived_inhibited_actions_basis.get_state_action_index(mapped_state, 0)

                if not self.derived_inhibited_actions_arr[inhibited_index]:

                    non_term_prim, reachable_subtasks  = self.get_all_reachable_non_term_actions(state, "Root", agent_id)
                    term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
                    self.derived_inhibited_actions_arr[inhibited_index] = (term_prim, reachable_subtasks)
                else:
                    [term_prim, _] = self.derived_inhibited_actions_arr[inhibited_index]

                for inhibited_action in term_prim:
                    prim_action = self.primitive_action_map[inhibited_action]
                    inhibited_samples = [mapped_state, np.asarray([prim_action]), self.min_reward, s_prime]
                    self.current_inhibited_abstracted_samples[agent_id].append(inhibited_samples)

    def generate_inhibited_samples(self, observation, agent_id):
        root_pg = self.completion_function_pg[agent_id]['Root']
        if len(self.num_states) != len(observation):
            mapped_obs = self._map_table_indices(observation, root_pg.policy)
        else:
            mapped_obs = self.slice_observation(root_pg, observation, agent_id)

        inhibited_index = self.derived_inhibited_actions_basis.get_state_action_index(mapped_obs, 0)

        if not self.derived_inhibited_actions_arr[inhibited_index]:
            non_term_prim, reachable_subtasks = self.get_all_reachable_non_term_actions(observation, "Root", agent_id)
            term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
            self.derived_inhibited_actions_arr[inhibited_index] = (term_prim, reachable_subtasks)
        else:
            [term_prim, _] = self.derived_inhibited_actions_arr[inhibited_index]

        for inhibited_action in term_prim:
            prim_action = self.primitive_action_map[inhibited_action]
            inhibited_samples = [mapped_obs, np.asarray([prim_action]), self.min_reward, None]
            self.current_inhibited_samples[agent_id].append(inhibited_samples)

    def get_term_primitives(self, observation, agent_id):

        mapped_state = self.map_obs(observation, agent_id)
        inhibited_index = self.derived_inhibited_actions_basis.get_state_action_index(mapped_state, 0)

        if not self.derived_inhibited_actions_arr[inhibited_index]:
            non_term_prim, reachable_subtasks = self.get_all_reachable_non_term_actions(observation, "Root", agent_id)
            term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
            self.derived_inhibited_actions_arr[inhibited_index] = (term_prim, reachable_subtasks)
        else:
            [term_prim, reachable_subtasks] = self.derived_inhibited_actions_arr[inhibited_index]

        return term_prim, reachable_subtasks

    def _add_derived_action_prim_children(self, observation, parent, non_term_prim, agent_id=0):
        """
        Returns the all primitive actions reachable from parent
        Adds the terminated actions to a global list
        """
        for child in self.get_derived_action_children(parent):
            if self.derived_is_primitive(child) and child not in non_term_prim:
                non_term_prim.append(child)
            elif not self.derived_is_terminated(child, observation, agent_id):
                non_term_prim = self._add_derived_action_prim_children(observation, child, non_term_prim, agent_id)
            elif not self.derived_is_primitive(child):
                if not self.term_subtasks:
                    self.term_subtasks[agent_id] = []
                self.term_subtasks[agent_id].append(child)
        return non_term_prim

    def get_all_reachable_non_term_actions(self, observation, parent, agent_id):
        """
        Entry point for helper method to find all of the primitive actions that are not terminated
        """
        non_term_prim, reachable_subtasks = self._add_derived_non_term_action(observation, parent, [], [], agent_id)
        return non_term_prim, reachable_subtasks

    def _add_derived_non_term_action(self, observation, parent, non_term_prim, reachable_subtask, agent_id=0):
        """
        Returns the all primitive actions reachable from parent
        Adds the terminated actions to a global list
        """
        for child in self.get_derived_action_children(parent):
            if self.derived_is_primitive(child) and child not in non_term_prim:
                non_term_prim.append(child)
            elif not self.derived_is_terminated(child, observation, agent_id):
                reachable_subtask.append(child)
                non_term_prim, reachable_subtask = self._add_derived_non_term_action(observation, child, non_term_prim, reachable_subtask, agent_id)
            elif not self.derived_is_primitive(child):
                if not self.term_subtasks:
                    self.term_subtasks[agent_id] = []
                self.term_subtasks[agent_id].append(child)
        return non_term_prim, reachable_subtask

    def get_derived_action_children(self, current_node):
        return self.derived_samples_hierarchy.action_hierarchy.get_children(current_node)

    def derived_is_primitive(self, current_node) -> bool:
        if not self.derived_samples_hierarchy.action_hierarchy.get_children(current_node):
            return True
        else:
            return False

    def derived_is_terminated(self, current_node: str, observation: np.array, agent_id: int):
        """
        Returns True if an subtask is terminated

        :param current_node: current subtask
        :param observation: current state
        :param agent_id: agent id
        :return: True if current node is terminated
        """
        # Primitive action always terminates
        if self.derived_is_primitive(current_node):
            return True
        else:
            termination_pred = self.derived_samples_hierarchy.action_hierarchy.get_termination_predicate(current_node)
            cf = self.derived_policy_map[agent_id][current_node]
            obs_domain = cf.policy.domain_obs
            obs = self.slice_observation(cf, observation, agent_id)
            additional_vars = self.derived_make_additional_var(termination_pred, current_node, agent_id, observation)
            terminated = predicates.evaluate_predicate(termination_pred, obs_domain, obs, additional_vars)
            return terminated

    def derived_make_additional_var(self, termination_pred, current_node, agent_id, full_observation):
        """
        Creates additional variable field for checking predicates
        For pseudo-rewards or termination predicates

        :param termination_pred: Termination Predicate
        :param current_node: Current subtask
        :param agent_id: agent id
        :return: additional variable field
        """
        additional_var = {}
        for term_p in termination_pred:
            if 'agent_num' in term_p:
                additional_var['agent_num'] = agent_id
            if 'target' in term_p:
                additional_var['target'] = self.derived_samples_hierarchy.action_hierarchy.ground_var_map[current_node]['target']

            if 'gold_in_region' in term_p:
                additional_var['gold_in_region'] = full_observation[self.agent_class_observation_domains[0].
                                                                        index_for_name("gold_in_region")][0]
            if 'wood_in_region' in term_p:
                additional_var['wood_in_region'] = full_observation[self.agent_class_observation_domains[0].
                                                                        index_for_name("wood_in_region")][0]

            if 'meet_gold_requirement' in term_p:
                additional_var['meet_gold_requirement'] = full_observation[self.agent_class_observation_domains[0].
                                                                        index_for_name("meet_gold_requirement")][0]

            if 'meet_wood_requirement' in term_p:
                additional_var['meet_wood_requirement'] = full_observation[self.agent_class_observation_domains[0].
                                                                        index_for_name("meet_wood_requirement")][0]

            # TODO Check state variables for exta info

        return additional_var

#----------------------------------------------------------------------------
# THis is the code for generating flattened hirarchies


def flatten(subtask: str, hierarchy: dict):
    '''

    :param subtask: subtask that is being flattened (i.e removed from the hierarchy)
    :param hierarchy: original hierarchy (i.e derived)
    :return:
    '''

    st_parents = hierarchy[subtask]['parents']
    st_children = hierarchy[subtask]['children']

    for parent in st_parents:
        if subtask in hierarchy[parent]['children']:
            hierarchy[parent]['children'].pop(subtask)

        for child in st_children:
            # child is not a parameterized action
            if child in hierarchy:
                if subtask in hierarchy[child]['parents']:
                    hierarchy[child]['parents'].remove(subtask)
                if child not in hierarchy[parent]['children']:
                    hierarchy[parent]['children'][child] = hierarchy[subtask]['children'][child]
                elif hierarchy[subtask]['children'][child] not in hierarchy[parent]['children'][child]:
                    hierarchy[parent]['children'][child].extend(hierarchy[subtask]['children'][child])
                if parent not in hierarchy[child]['parents']:
                    hierarchy[child]['parents'].append(parent)

    hierarchy.pop(subtask)
    return hierarchy


def top_down_flatten(hierarchy, iterations=1, keep_navigate=False):
    assert 'Root' in hierarchy, "Root must be in hierarchy"
    original_hierarchy = copy.deepcopy(hierarchy)

    for i in range(iterations):
        depths_map = {}

        max_child = None
        max_depth = -1
        # Get longest path...
        for child in copy.deepcopy(original_hierarchy['Root']['children']):
            depths_map[child] = get_path_length(original_hierarchy, child)
            if depths_map[child] > max_depth:
                max_child = [child]
                max_depth = depths_map[child]
            elif depths_map[child] == max_depth:
                max_child.append(child)

        if max_child is not None:
            for child in max_child:
                if not hierarchy[child]['primitive'] and (not keep_navigate or (child != 'Navigate' and 'Navigate' not in child)):
                    original_hierarchy = flatten(child, original_hierarchy)

    return original_hierarchy


def get_path_length(hierarchy, subtask):
    # Recursively get path length
    if  hierarchy[subtask]['primitive']:
        return 1
    else:
        return 1 + max([get_path_length(hierarchy, child) for child in hierarchy[subtask]['children']])


def bottom_up_flatten(hierarchy, iterations=3, keep_navigate=False):
    assert 'Root' in hierarchy, "Root must be in hierarchy"
    original_hierarchy = copy.deepcopy(hierarchy)
    for i in range(iterations):
        depths_list = get_depths(original_hierarchy)
        for child in depths_list[-2]:
            if not hierarchy[child]['primitive'] and (not keep_navigate or (child != 'Navigate' and 'Navigate' not in child)):
                original_hierarchy = flatten(child, original_hierarchy)
    return original_hierarchy


def get_depths(hierarchy: dict) -> list:
    dl = [["Root"]]
    all_primitives = False

    while not all_primitives:
        c_set = set()
        all_primitives = True
        for st in dl[-1]:
            if not hierarchy[st]['primitive']:
                for child in hierarchy[st]['children']:
                    if not hierarchy[child]['primitive']:
                        all_primitives = False
                    c_set.add(child)
        dl.append(list(c_set))
    return dl
