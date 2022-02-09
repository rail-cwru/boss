from typing import Dict, List, Tuple, Any

import numpy as np
import copy as copy
from domain.ActionDomain import ActionDomain
from domain.observation import ObservationDomain
from config import AbstractModuleFrame, ConfigItemDesc
import itertools
from domain.features import CoordinateFeature, DiscreteFeature, BinaryFeature
from agentsystem.HierarchicalSystem import HierarchicalSystem
from config import Config, checks
from policy.function_approximator.basis_function.ExactBasis import ExactBasis
from domain.conversion import FeatureConversions
import pickle as pkl

class PolledSampler(HierarchicalSystem, AbstractModuleFrame):
    """
    An agent system designed for polled, hierarchical sampling for offline learning
    Designed for offline sampling only, not intended for use as an agentSystem for online learning
    @author Eric Miller
    @contact edm54@case.edu
    """
    def get_class_config() -> List[ConfigItemDesc]:
        return[
            ConfigItemDesc('collect_abstract', checks.boolean, 'Collect abstract samples', default=True, optional=True),
            ConfigItemDesc('collect_inhibited', checks.boolean,
                           'Collect inhibited samples', default=True, optional=True),
            ConfigItemDesc('min_reward', checks.negative_integer,
                           'Minimum Reward for Inhibited Samples',
                           default=-20, optional=True),
            ConfigItemDesc('optimistic', checks.boolean, 'Use balanced wandering', default=False, optional=True),
            ConfigItemDesc('use_primitive_dist',
                           checks.boolean,
                           'Use the number of times a primitive action has been sampled, regardless of parent',
                           default=False,
                           optional=True),
            ConfigItemDesc('save_target',
                           checks.boolean,
                           'Save distribution as target distro for kl divergence',
                           default=False,
                           optional=True),
            ConfigItemDesc('save_name', checks.string, 'Name for saving target distribution for future testing',
                           default='chair_target.list', optional=True),
        ]

    def __init__(self, agent_id_class_map: Dict[int, int], agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain], auxiliary_data: Dict[str, Any],
                 config: Config):

        verbose = False if config.sampler.name == 'BOSS' else True

        self.term_subtasks = {}
        super(PolledSampler, self).__init__(agent_id_class_map, agent_class_action_domains,
                                            agent_class_observation_domains, auxiliary_data, config)

        self.min_reward = config.sampler.min_reward if hasattr(config.sampler, 'min_reward') else -100
        self.save_target = config.sampler.save_target if hasattr(config.sampler, 'save_target') else False
        self.save_name = config.sampler.save_name if hasattr(config.sampler, 'save_name') else 'chair_target.list'

        self.collect_inhibited = config.sampler.collect_inhibited
        self.collect_abstract = config.sampler.collect_abstract
        self.use_primitive_distribution = config.sampler.use_primitive_dist if hasattr(config.sampler, 'use_primitive_dist') else False

        self.optimistic = config.sampler.optimistic if hasattr(config.sampler, 'optimistic') else False
        if verbose:
            print("Use balanced wandering:", self.optimistic)

            if self.save_target:
                print("Save Target")

        self.inhibited_abstract_samples = []

        self._init_derived()
        self.per_sample_derived = []
        self.first_action = True
        self.num_states = [domain_item.num_values() for domain_item in
                           agent_class_observation_domains[agent_id_class_map[0]].items]

        self.num_actions = agent_class_action_domains[0].full_range
        self.temperature = config.sampler.temperature if hasattr(config.sampler, "temperature") else 1.0
        self._create_state_visits()

        if hasattr(config, 'display_distribution'):
            self.num_actions = agent_class_action_domains[0].full_range
            self._create_state_visits()
            self.display_distribution = config.display_distribution
            self.add_derived = True
            self.distributions = []
        else:
            self.display_distribution = False
            self.add_derived = False

        self.kl_div = False
        save_target = False

        if config.kl_divergence:
            self.kl_div = True
            print('Preparing target distribution!')
            if save_target:
                self.target_distribution = self.get_target_dist()
                self.save_target_dist()
            else:
                self.load_target_dist()

            self.num_actions = agent_class_action_domains[0].full_range
            self._create_state_visits()
            self.distributions = []

        self.inhibited_actions_basis = ExactBasis(np.asarray(self.num_states), 1)
        self.inhibited_actions_arr = [None for _ in range(self.inhibited_actions_basis.size())]
        if verbose:
            print("Collect Inhibited:", self.collect_inhibited, "Collect Abstract:", self.collect_abstract)

    def _create_state_visits(self):
        """
        Initialize state visits arrays for both the general policy and for each subtask including primitives
        """

        self.state_action_basis = ExactBasis(np.asarray(self.num_states), self.num_actions)
        self.state_action_visits = np.zeros(self.state_action_basis.size(), np.int64)

        self.state_action_basis_map = {}
        self.state_action_visits_map = {}

        for name, pg in self.completion_function_pg[0].items():
            # if not self.is_primitive(name) or self.use_primitive_distribution:
            if self.collect_abstract:
                # Use abstracted state space
                obs_domain = pg.policy.domain_obs
                num_states = [domain_item.num_values() for domain_item in
                              obs_domain.items]
            else:
                # Use entire state since abstraction not being used
                num_states = self.num_states

            num_actions = pg.policy.domain_act.full_range
            self.state_action_basis_map[name] = ExactBasis(np.asarray(num_states), num_actions)
            self.state_action_visits_map[name] = np.zeros(self.state_action_basis_map[name].size(), np.int64)

    def get_derived_samples(self, last_obs, sucessful_traj):
        """
        Returns the derived samples
        :return:
        """
        if not sucessful_traj:
            self.add_s_prime(last_obs[0], 0)
        else:
            self.add_s_prime(None, 0)
        return self.inhibited_samples[0], self.abstracted_samples[0]

    def add_s_prime(self, s_prime, agent_id):
        """
        Adds the next state to the derived samples. Also adds the reward to the abstracted samples

        :param s_prime: next state
        :param agent_id: agent id to update
        :return:
        """
        single_episode_derived = []
        if s_prime is not None:
            mapped_s_prime = self._map_table_indices(s_prime[:], self.completion_function_pg[agent_id]["Root"].policy)
        else:
            mapped_s_prime = None

        for inhibited_a in self.current_inhibited_samples[agent_id]:
            inhibited_a[-1] = mapped_s_prime
            self.inhibited_samples[agent_id].append(inhibited_a[:])
            single_episode_derived.append(inhibited_a[:])

        for abstract_s in self.current_abstracted_samples[agent_id]:
            action = self.action_stack_dict[agent_id][-1]
            reward = self.current_reward_dict[agent_id][action]
            if s_prime is None:
                abstract_s[-1] = mapped_s_prime
            else:
                updated_s_prime = abstract_s[-1]
                ind = np.where(updated_s_prime == None)
                updated_s_prime[ind] = s_prime[ind]
                abstract_s[-1] = self._map_table_indices(updated_s_prime[:], self.completion_function_pg[agent_id]["Root"].policy)

            abstract_s[-2] = reward
            self.abstracted_samples[agent_id].append(abstract_s[:])
            single_episode_derived.append(abstract_s[:])

        for inhibited_abstracted_s in self.current_inhibited_abstracted_samples[agent_id]:
            if s_prime is None:
                inhibited_abstracted_s[-1] = s_prime
            else:
                updated_s_prime = inhibited_abstracted_s[-1]
                ind = np.where(updated_s_prime == None)
                updated_s_prime[ind] = s_prime[ind]
                inhibited_abstracted_s[-1] = self._map_table_indices(updated_s_prime[:], self.completion_function_pg[agent_id]["Root"].policy)

            self.inhibited_samples[agent_id].append(inhibited_abstracted_s[:])
            single_episode_derived.append(inhibited_abstracted_s[:])
            self.inhibited_abstract_samples.append(inhibited_abstracted_s[:])

        self.current_inhibited_samples[agent_id] = []
        self.current_abstracted_samples[agent_id] = []
        self.current_inhibited_abstracted_samples[agent_id] = []
        self.per_sample_derived.append(single_episode_derived)

    def _prepare_derived(self, observation, agent_id, add_s_prime=True):
        if not self.first_action and add_s_prime:
            self.add_s_prime(observation, agent_id)

        self.clear_terminated_actions(agent_id)
        self.child_terminated_dict[agent_id] = {}
        self.term_subtasks[agent_id] = []

    # This method uses the current node from the global stack for each agent
    def get_actions(self, observations: Dict[int, np.ndarray], use_max=False, add_s_prime=True) -> Dict[int, np.ndarray]:
        """
        Implementation of the MAXQ action selection
        Returns a primitive action for each agent using an observation. Start at current non-primitive action in the
        agent action stack, traverses down hierarchy through non-primitives, storing each in a stack.
        Will only call non-terminated actions (for ex, cannot call Put without passenger)

        This implementation will return a random action at each subtask`
        :param observations: Current Observation from Environment
        :param use_max: if true, agent use a greedy policy
        :return: dict mapping Agent to list of primitive action
        """
        all_actions = {}

        for agent_id, observation in observations.items():
            self._prepare_derived(observation, agent_id, add_s_prime)

            current_node = 'Root'
            self.action_stack_dict[agent_id] = ['Root']

            # recurse on hierarchy until primitive action
            while not self.is_primitive(current_node):
                policy_group = self.completion_function_pg[agent_id][current_node]

                # Get non-teminated action
                if self.optimistic:
                    policy_group = self.completion_function_pg[agent_id][current_node]
                    basis = self.state_action_basis_map[current_node]
                    previous_node = copy.copy(current_node)
                    abstracted_observation = self.abstract_state(observation, current_node, agent_id)

                    if len(self.num_states) != len(observation):
                        abstracted_observation = self._map_table_indices(observation, policy_group.policy)

                    current_node, chosen_act = self.get_non_term_action_optimistic(observation,
                                                                                   current_node,
                                                                                   agent_id,
                                                                                   use_max=use_max)

                    self.state_action_visits_map[previous_node][basis.get_state_action_index(abstracted_observation,
                                                                                             chosen_act[0])] += 1
                else:
                    current_node, chosen_act = self.get_non_term_action(observation, current_node, agent_id,
                                                                        use_max=use_max)

                abstracted_observation = self.slice_observation(policy_group, observation, agent_id)

                # For storing trajectory
                self._update_oar(agent_id, policy_group.pg_id, abstracted_observation, chosen_act, 0)

                # initialize sequence
                self.action_sequence_dict[agent_id][current_node] = []
                self.action_stack_dict[agent_id].append(current_node)

            # Store primitive action for state to execute
            primitive_action = np.asarray([self.primitive_action_map[current_node]])
            all_actions[agent_id] = primitive_action

            if hasattr(self, "derived_policy_map"):
                policy_group = self.derived_policy_map[agent_id][current_node]
            else:
                policy_group = self.completion_function_pg[agent_id][current_node]

            if not self.display_distribution:
                # Use child PG since we can set SV that way
                self._collect_derived(observation, policy_group, primitive_action, agent_id, current_node)

            if hasattr(self, "derived_policy_map"):
                policy_group = self.completion_function_pg[agent_id][current_node]

            # append state to primitive action
            self.action_sequence_dict[agent_id][current_node].append(observation)
            abstracted_observation = self.slice_observation(policy_group, observation, agent_id)

            # Each primitive has own value function so it must be [0]
            self._update_oar(agent_id, current_node, abstracted_observation, [0], 0)
            self.first_action = False

            if self.display_distribution or self.kl_div or self.save_target:
                # UPDATE
                root_pg = self.completion_function_pg[agent_id]["Root"]
                mapped_obs = self._map_table_indices(observation, root_pg.policy)
                index = self.state_action_basis.get_state_action_index(mapped_obs, primitive_action[0])
                self.state_action_visits[index] += 1

        return all_actions

    def get_non_term_action_optimistic(self, observation, parent, agent_id, use_max=False):
        """
        Selects an action from the children of parent
        Checks to make sure the action is not already terminated (i.e navigate_0_0 in 0,0)

        :param observation: Current state (non-abstracted)
        :param parent: Current node in hierarchy
        :param agent_id: Agent id
        :param use_max: Boolean dictating use of greedy policy
        :return: name of action, index of action
        """
        # Holds set of terminated children to avoid selecting them
        mapped_observation = self.abstract_state(observation, parent, agent_id)
        basis = self.state_action_basis_map[parent]
        sa_visits = self.state_action_visits_map[parent]

        term_child = set()
        current_node = parent

        policy_group = self.completion_function_pg[agent_id][current_node]
        action_values = list(policy_group.policy.domain_act.get_action_range()[0])

        term_prim, reachable_subtasks = self.get_term_primitives(observation, agent_id)
        non_term_actions = []

        nta = []

        for action in action_values:
            current_node = self.int_to_action(action, parent)
            is_terminated = self.is_primitive(current_node) or current_node not in reachable_subtasks
            # self.is_terminated(current_node, observation, agent_id)
            if not self.is_primitive(current_node) and is_terminated:
                self.term_subtasks[agent_id].append(current_node)
                term_child.add(action)
            else:
                # Action must either be a non-terminated subtask or is a reachable primitve
                if (not self.is_primitive(current_node) and current_node in reachable_subtasks) or (
                        self.is_primitive(current_node) and current_node not in term_prim):
                    non_term_actions.append(action)
                    nta.append(current_node)

        if len(non_term_actions) == 0:
            raise ValueError('No Actions Left')
        elif len(non_term_actions) == 1:
            return self.int_to_action(non_term_actions[0], parent), non_term_actions
        else:

            sa_index_list = [basis.get_state_action_index(mapped_observation, action) for action in non_term_actions]
            num_visits = np.asarray([sa_visits[index] for index in sa_index_list])

            # Since it does not really matter which subtask called the primitive action,
            # due to shared value functions, use the SA count of primitives whenever applicable
            if self.use_primitive_distribution:
                # 1) Check for primtive children
                for ind, child in enumerate(self.get_action_children(parent)):
                    if self.is_primitive(child) and ind in non_term_actions:
                        # Action always 1
                        sa_index = self.state_action_basis_map[child].get_state_action_index(mapped_observation, 0)

                        # 2) Get num visits for the primitive children
                        num_prim_visits = self.state_action_visits_map[child][sa_index]

                        # No terminated actions, so sub right on in
                        if len(non_term_actions) == len(action_values):
                            # 3) Swap into num_visits
                            num_visits[ind] = num_prim_visits
                        else:
                            # Need to find the actual index of the prim action to sub in
                            actual_ind = np.where(num_visits == ind)
                            num_visits[actual_ind] = num_prim_visits

            if num_visits.max() == 0:
                chosen_act = [np.random.choice(non_term_actions)]
            else:
                # Select min with random tiebreak
                chosen_act = [np.random.choice([i for ind, i in enumerate(non_term_actions) if num_visits[ind] == num_visits.min()])]

            current_node = self.int_to_action(chosen_act[0], parent)

        return current_node, chosen_act

    def get_non_term_action(self, observation, parent, agent_id, use_max=False):
        """
        Selects an action from the children of parent
        Checks to make sure the action is not already terminated (i.e navigate_0_0 in 0,0)

        :param observation: Current state (non-abstracted)
        :param parent: Current node in hierarchy
        :param agent_id: Agent id
        :param use_max: Boolean dictating use of greedy policy
        :return: name of action, index of action
        """
        term_child = set()

        # Holds set of terminated children to avoid selecting them
        current_node = parent

        policy_group = self.completion_function_pg[agent_id][current_node]
        action_values = list(policy_group.policy.domain_act.get_action_range()[0])

        term_prim, reachable_subtasks = self.get_term_primitives(observation, agent_id)
        non_term_actions = []

        for action in action_values:
            current_node = self.int_to_action(action, parent)
            is_terminated = self.is_primitive(current_node) or current_node not in reachable_subtasks
                            # self.is_terminated(current_node, observation, agent_id)
            if not self.is_primitive(current_node) and is_terminated:
                self.term_subtasks[agent_id].append(current_node)
                term_child.add(action)
            else:
                # Action must either be a non-terminated subtask or is a reachable primitve
                if (not self.is_primitive(current_node) and current_node in reachable_subtasks) or (
                        self.is_primitive(current_node) and current_node not in term_prim):

                  non_term_actions.append(action)

        if len(non_term_actions) == 0:
            raise ValueError('No Actions Left')

        chosen_act = [np.random.choice(non_term_actions)]
        current_node = self.int_to_action(chosen_act[0], parent)
        term_child.add(chosen_act[0])
        return current_node, chosen_act

    def _collect_derived(self, observation, policy_group, primitive_action, agent_id, current_node=None):
        if self.collect_inhibited:
            self.generate_inhibited_samples(observation, agent_id)
        if self.collect_abstract:
            self.generate_abstract_samples(observation, policy_group, agent_id, primitive_action, current_node=current_node)

    def generate_inhibited_from_abstract(self, agent_id):
        for abstract_sample in self.current_abstracted_samples[agent_id]:
            state = abstract_sample[0]
            s_prime = abstract_sample[-1]

            if len(self.num_states) != len(state):
                mapped_state = self._map_table_indices(state[:], self.completion_function_pg[agent_id]["Root"].policy)
            else:
                mapped_state = state[:]

            inhibited_index = self.inhibited_actions_basis.get_state_action_index(mapped_state, 0)

            # Uses the previously found non-reachable primitives if this state has been tested
            if self.inhibited_actions_arr[inhibited_index] is None:
                non_term_prim = self.get_all_non_term_primitives(state, "Root", agent_id)
                term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
                self.inhibited_actions_arr[inhibited_index] = term_prim

            else:
                term_prim = self.inhibited_actions_arr[inhibited_index]

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
        # if self.display_distribution and self.add_derived:
        root_pg = self.completion_function_pg[agent_id]['Root']
        state_list, s_prime_list = self.get_irrelevant_states(observation, policy_group, agent_id)
        for ind, state in enumerate(state_list):

            if len(self.num_states) != len(observation):
                mapped_obs = self._map_table_indices(state, root_pg.policy)
            else:
                mapped_obs = self.slice_observation(root_pg, state, agent_id)

            sample = [mapped_obs, np.asarray([primitive_action]), None, s_prime_list[ind]]
            self.current_abstracted_samples[agent_id].append(sample)

            # generate inhibited from abstract
            if self.collect_inhibited and self.collect_abstract:
                inhibited_index = self.inhibited_actions_basis.get_state_action_index(mapped_obs, 0)

                # Uses the previously found non-reachable primitives if this state has been tested
                if self.inhibited_actions_arr[inhibited_index] is None:
                    non_term_prim, reachable_subtasks = self.get_all_reachable_non_term_actions(state, "Root",
                                                                                                agent_id)
                    # print(non_term_prim, reachable_subtasks, mapped_obs, inhibited_index)
                    term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
                    self.inhibited_actions_arr[inhibited_index] = (term_prim, reachable_subtasks)
                else:
                    [term_prim, _] = self.inhibited_actions_arr[inhibited_index]

                for inhibited_action in term_prim:
                    prim_action = self.primitive_action_map[inhibited_action]
                    inhibited_samples = [mapped_obs, np.asarray([prim_action]), self.min_reward, s_prime_list[ind]]
                    self.current_inhibited_abstracted_samples[agent_id].append(inhibited_samples)

            if self.display_distribution and self.add_derived:
                index = self.state_action_basis.get_state_action_index(mapped_obs, primitive_action)
                self.state_action_visits[index] += 1

    def get_irrelevant_states(self, observation, policy_group, agent_id):
        '''
        Finds the state variables that are irrelevant based on hierarchy
        :param observation: Current (non-abstracted) observation
        :param policy_group: policy group of current node (parent of primitive/primitive)
        :param agent_id:
        :return: Abstract states and sprimes based on the input observation
        '''
        agent_class = self.agent_id_class_map[agent_id]
        root_pg = self.policy_groups[agent_class]
        full_obs_domain = root_pg.policy.domain_obs
        item_slices = []
        item_values = []
        # Find the irrelevant state variables
        for item in full_obs_domain.items:
            if item not in policy_group.policy.domain_obs.items:
                item_vals = self._get_feature_range(item)
                domain_slice = self.agent_slice_dict[agent_class][item.name]
                item_slices.append(domain_slice)
                item_values.append(item_vals)

        # Get all combinations of state variables
        irrelevant_state_value_pairs = list(itertools.product(*item_values))
        states = []
        s_primes = []

        # Change just the irrelevant values and append to list
        for state_pair in irrelevant_state_value_pairs:
            state = copy.copy(observation)
            s_prime = np.asarray([None] * len(observation))
            for ind, value in enumerate(state_pair):
                domain_slice = item_slices[ind]
                state[domain_slice] = value
                s_prime[domain_slice] = value
            if not np.all(state == observation):
                states.append(copy.deepcopy(state))
                s_primes.append(copy.deepcopy(s_prime))

        return states, s_primes

    def generate_inhibited_samples(self, observation, agent_id):
        """
        Generates inhibited action samples, or samples that set the reward of taking an illegal action (from hierarchy)
        to a large negative number

        Only does it for the current observation, not for a list of samples as proposed in Devin's thesis

        :param observation: current observation
        :param agent_id: current agent_id
        :return:
        """
        root_pg = self.completion_function_pg[agent_id]['Root']
        if len(self.num_states) != len(observation):
            mapped_obs = self._map_table_indices(observation, root_pg.policy)
        else:
            mapped_obs = self.slice_observation(root_pg, observation, agent_id)

        inhibited_index = self.inhibited_actions_basis.get_state_action_index(mapped_obs, 0)

        # Uses the previously found non-reachable primitives if this state has been tested
        if self.inhibited_actions_arr[inhibited_index] is None:
            non_term_prim, reachable_subtasks = self.get_all_reachable_non_term_actions(observation, "Root",
                                                                                        agent_id)

            term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
            self.inhibited_actions_arr[inhibited_index] = (term_prim, reachable_subtasks)
        else:
            [term_prim, _] = self.inhibited_actions_arr[inhibited_index]

        blocked_acts = term_prim

        for inhibited_action in blocked_acts:
            prim_action = self.primitive_action_map[inhibited_action]
            inhibited_samples = [mapped_obs, np.asarray([prim_action]), self.min_reward, None]
            self.current_inhibited_samples[agent_id].append(inhibited_samples)

            if self.display_distribution and self.add_derived:
                index = self.state_action_basis.get_state_action_index(mapped_obs, prim_action)
                self.state_action_visits[index] += 1

    def reset(self):
        '''
        Resets sampler for next episode
        :return:
        '''
        super(PolledSampler, self).reset()
        self.first_action = True
        self._init_derived()

        if (self.display_distribution or self.kl_div) and sum(self.state_action_visits) > 0:
            self.distributions.append(self.state_action_visits/sum(self.state_action_visits))

    def check_all_agent_termination(self, observations: np.array):
        pass

    def _map_table_indices(self, states: np.ndarray, policy) -> Tuple[int, ...]:
        """
        Map a single state from native representation to table index.
        :param states: State in native representation
        :return: Table index
        """
        mapped_state = []
        for feature in policy.domain_obs.items:
            # Get individual domain item state and map it to the requested interpretation
            domain_state = policy.domain_obs.get_item_view_by_item(states, feature)
            domain_mapped_state = FeatureConversions.as_index(feature, domain_state)
            mapped_state.append(domain_mapped_state)
        return np.asarray(mapped_state)

    def _get_feature_range(self, item):
        """
        Returns the range of values for a feature of different types
        Useful for abstract samples
        """
        if isinstance(item, DiscreteFeature):
            item_vals = [i for i in range(item.range.start, item.range.stop)]

        elif isinstance(item, CoordinateFeature):
            if item.sparse_values:
                item_vals = item.sparse_values
            else:
                item_vals = [i for i in range(item.lower, item.upper)]
        elif isinstance(item, BinaryFeature):
            item_vals = [0, 1]
        else:
            raise ValueError("Only Discrete, Binary and Coordinate Features Supported")
        return item_vals

    def get_all_non_term_primitives(self, observation, parent, agent_id):
        """
        Entry point for helper method to find all of the primitive actions that are not terminated
        """
        non_term_prim = self._add_action_prim_children(observation, parent, [], agent_id)
        return non_term_prim

    def _add_action_prim_children(self, observation, parent, non_term_prim, agent_id=0):
        """
        Returns the all primitive actions reachable from parent
        Adds the terminated actions to a global list
        """
        for child in self.get_action_children(parent):
            if self.is_primitive(child) and child not in non_term_prim:
                non_term_prim.append(child)
            elif not self.is_terminated(child, observation, agent_id):
                non_term_prim = self._add_action_prim_children(observation, child, non_term_prim, agent_id)
            elif not self.is_primitive(child):
                self.term_subtasks[agent_id].append(child)
        return non_term_prim

    def _init_derived(self):
        """
        Initialize the derived samples dictionaries
        """

        # inhibited actions go into the current inhibited samples list, then
        self.inhibited_samples = {}
        self.current_inhibited_samples = {}
        self.abstracted_samples = {}
        self.current_abstracted_samples = {}
        self.current_inhibited_abstracted_samples = {}

        for agent_id in self.agent_id_class_map.keys():
            self.inhibited_samples[agent_id] = []
            self.current_inhibited_samples[agent_id] = []
            self.abstracted_samples[agent_id] = []
            self.current_abstracted_samples[agent_id] = []
            self.current_inhibited_abstracted_samples[agent_id] = []

    def get_per_sample_derived(self):
        return self.per_sample_derived

    def _is_taxi_world(self):
        return "Get" in self.get_action_children("Root") or "Pickup" in self.get_action_children("Root")

    def _get_action_dist(self, state):
        current_node = 'Root'
        if self.is_terminated(current_node, state, 0):
            return np.zeros(self.num_actions)
        probability_map = self._get_prim_dist(state, current_node)
        return probability_map

    def _get_prim_dist(self, state, current_node):
        non_term_children, total_children = self._get_num_non_term_children(current_node, state)
        probability_map = {}

        for child in self.get_action_children(current_node):
            if self.is_primitive(child):
                probability_map[child] = 1 / non_term_children
            elif not self.is_terminated(child, state, 0):
                p_map = self._get_prim_dist(state, child)

                for key, value in p_map.items():
                    prob = value * 1/non_term_children
                    if key in probability_map:
                        probability_map[key] += prob
                    else:
                        probability_map[key] = prob

        return probability_map

    def load_target_dist(self):

        file_name = self.save_name

        with open(file_name, 'rb') as target_dist_file:
            self.target_distribution = pkl.load(target_dist_file)
        print('Loaded', file_name)

    def save_target_dist(self):
        file_name = self.save_name
        with open(file_name, 'wb') as target_dist_file:
            pkl.dump(self.target_distribution, target_dist_file)
        print('Saved', file_name)

    def save_dist(self, distribution):

        file_name = self.save_name
        self.target_distribution = distribution

        with open(file_name, 'wb') as target_dist_file:
            pkl.dump(distribution, target_dist_file)
        print('Saved', file_name)

    def _get_num_non_term_children(self, current_node, state):
        count = 0
        all_children = self.get_action_children(current_node)
        for child in all_children:
            if self.is_primitive(child) or not self.is_terminated(child, state, 0):
                count += 1

        return count, len(all_children)

    def get_distribution(self):
        return self.state_action_visits/sum(self.state_action_visits)

    def abstract_state(self, observation, current_node, agent_id):
        if self.collect_abstract:
            pg = self.completion_function_pg[agent_id][current_node]
            if len(self.num_states) != len(observation):
                abstracted_observation = self._map_table_indices(observation, pg.policy)
            else:
                abstracted_observation = self.slice_observation(pg, observation, agent_id)
        else:
            pg = self.completion_function_pg[agent_id]["Root"]
            if len(self.num_states) != len(observation):
                abstracted_observation = self._map_table_indices(observation, pg.policy)
            else:
                abstracted_observation = self.slice_observation(pg, observation, agent_id)
        return abstracted_observation

    def get_term_primitives(self, observation, agent_id):

        mapped_state = self.map_obs(observation, agent_id)
        inhibited_index = self.inhibited_actions_basis.get_state_action_index(mapped_state, 0)

        if not self.inhibited_actions_arr[inhibited_index]:
            non_term_prim, reachable_subtasks = self.get_all_reachable_non_term_actions(observation, "Root", agent_id)
            term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
            self.inhibited_actions_arr[inhibited_index] = (term_prim, reachable_subtasks)
        else:
            [term_prim, reachable_subtasks] = self.inhibited_actions_arr[inhibited_index]

        return term_prim, reachable_subtasks

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
        for child in self.get_action_children(parent):
            if self.is_primitive(child) and child not in non_term_prim:
                non_term_prim.append(child)
            elif not self.is_terminated(child, observation, agent_id):
                reachable_subtask.append(child)
                non_term_prim, reachable_subtask = self._add_derived_non_term_action(observation, child, non_term_prim, reachable_subtask, agent_id)
            elif not self.is_primitive(child):
                if not self.term_subtasks:
                    self.term_subtasks[agent_id] = []
                self.term_subtasks[agent_id].append(child)
        return non_term_prim, reachable_subtask

    def map_obs(self, observation, agent_id):
        if len(self.num_states) != len(observation):
            mapped_state = self._map_table_indices(observation[:], self.policy_map[agent_id]["Root"])
        else:
            mapped_state = observation[:]

        return mapped_state

    def get_sample_probability(self, sample) -> float:
        probability_dict = {}
        probability_dict["Root"] = 1.0
        observation = sample[0]
        p = self.get_probs(observation)
        for k, v in self.primitive_action_map.items():
            if v == sample[1][0]:
                return p[k]

        raise ValueError('Action not found!')

    def get_probs(self, observation, agent_id=0):
        probability_map = {}
        probability_map["Root"] = 1.0
        term_prim, reachable_subtasks = self.get_term_primitives(observation, agent_id)
        nodes = ["Root"]

        while nodes:
            next_nodes = set()
            for node in nodes:
                children = self.get_non_term_children(node, reachable_subtasks, term_prim)
                for child in children:
                    probability_map[child] = probability_map.get(child, 0) + probability_map[node]/len(children)
                    if not self.is_primitive(child):
                        next_nodes.add(child)
            nodes = list(next_nodes)
        return probability_map

    def get_non_term_children(self,  parent, reachable_subtasks, term_prim, agent_id=0):
        policy_group = self.completion_function_pg[agent_id][parent]
        action_values = list(policy_group.policy.domain_act.get_action_range()[0])
        non_term_actions = []
        term_child = set()
        for action in action_values:
            current_node = self.int_to_action(action, parent)
            is_terminated = self.is_primitive(current_node) or current_node not in reachable_subtasks
            if not self.is_primitive(current_node) and is_terminated:
                term_child.add(action)
            else:
                # Action must either be a non-terminated subtask or is a reachable primitve
                if (not self.is_primitive(current_node) and current_node in reachable_subtasks) or (
                        self.is_primitive(current_node) and current_node not in term_prim):
                  non_term_actions.append(self.int_to_action(action, parent))

        return non_term_actions

    def get_depths(self, hierarchy, edges):
        dl = [["Root"]]
        all_primitives = False

        while not all_primitives:
            c_set = set()
            all_primitives = True
            print(dl[-1])
            for st in dl[-1]:
                if edges[st]:
                    print(edges[st])
                    for child in edges[st]:
                        if not hierarchy[child]['primitive']:
                            all_primitives = False
                        c_set.add(child)
            dl.append(list(c_set))
        return dl
