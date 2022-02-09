"""
Hierarchical AgentSystems that can support learning in a hierarchical environment
"""

from typing import Dict, List, Tuple, Union, Any

import numpy as np
import copy as copy
from common.trajectory import Trajectory
from domain.ActionDomain import ActionDomain
from domain.hierarchical_domain import HierarchicalActionDomain
from domain.observation import ObservationDomain
from . import AgentSystem
from .util import PolicyGroup
from config import Config, AbstractModuleFrame, ConfigItemDesc
from common.domain_transfer import DomainTransferMessage
from common.properties import Properties
from policy import Policy
from common.predicates import predicates
from algorithm import Algorithm
import operator

class HierarchicalSystem(AgentSystem, AbstractModuleFrame):
    """
    @author Eric Miller
    @contact edm54
    An Agent System where every individual agent represents an SMDP. Each agent learns a policy for a different set of
    actions. Actions higher in the hierarchy are non-primitive and are handled by other agents. Each action gets its own
    policy
    Note: agents that are not the actual agent interacting with environment are referred to as 'sub-agents' in comments
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return []

    @classmethod
    def properties(cls) -> Properties:
        return Properties()

    def __init__(self, agent_id_class_map: Dict[int, int], agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain], auxiliary_data: Dict[str, Any],
                 config: Config):

        # Store hierarchy
        self.hierarchy: HierarchicalActionDomain = None
        self.h_obs_domain: dict[str: ObservationDomain] = None

        if hasattr(auxiliary_data, 'hierarchy'):
            self.hierarchy = auxiliary_data.hierarchy
        else:
            print("No hierarchy found, results may be incorrect")

        if hasattr(auxiliary_data, 'joint_observation_domains'):
            self.h_obs_domain = auxiliary_data.joint_observation_domains

        # Store class map for multi-agents
        self.agent_id_class_map = agent_id_class_map

        # Maps an agent id to the current action stack which holds a string (Action's name)
        self.action_stack_dict = {}

        # Maps each agent to a dictionary mapping each sub-agent to its current observation, action, reward
        self.abstracted_obs_dict = {}
        self.current_action_dict = {}
        self.current_reward_dict = {}

        # maps an agent to a dict mapping each node in the stack to a sequence of states
        self.action_sequence_dict = {}

        # Maps agent id to a dictionary storing the hierarchical policy groups (for each agent in hierarchy)
        self.hierarchical_policy_dict = {}

        # Maps agents to a policy group id
        # Maps agents to a policy group id
        # This is for transfer domain only
        self.agent_policy_groups_map = {}

        # Holds action policies such that policies can be shared between different policy groups
        self.policy_map = {}
        self.pseudo_policy_map = {}

        # Maps an agent to a list of subtasks that had children terminate (for updating purposes)
        self.child_terminated_dict = {}

        # Initialize the dicts used for storing current observation and actions
        self._init_current_dicts(agent_id_class_map.keys())

        # For state abstractions
        self.agent_slice_dict = self.all_agent_slice_dict(agent_class_observation_domains)

        self.primitive_action_map = self.hierarchy.action_hierarchy.primitive_action_map

        # maps an agent to a policy group holding its completion function
        self.completion_function_pg = {}

        # Maps agent to pg holding its completion function calculated with pseudo-rewards
        self.pseudo_reward_cf_pg = {}

        self.means = []
        self.epi_means = []

        self.navigate_means = []
        self.primitive_means = []
        self.root_mean = []
        self.get_put_means = []
        self.putdown_means = []
        self.pickup_means = []
        super(HierarchicalSystem, self).__init__(agent_id_class_map, agent_class_action_domains,
                                                 agent_class_observation_domains, auxiliary_data, config)

    def _make_policy_groups(self, agent_id_class_map: Dict[int, int],
                            agent_class_action_domains: Dict[int, ActionDomain],
                            agent_class_observation_domains: Dict[int, ObservationDomain]) -> List[PolicyGroup]:
        """
        Makes all of the policy groups for agents and sub-agents. Stores sub-agent policy groups. Each agent gets its
        own policy group, but the policies are shared between agents and subtasks

        :param agent_id_class_map: Maps agent to class, agents with same class should be same policy
        :param agent_class_action_domains: Maps agent to Action Domain
        :param agent_class_observation_domains: Maps agent to observation domain
        :return: list of agent policy groups, indexed by agent id
        """
        policy_groups = []
        # Map agent class to policy
        pg_map = {}
        pseudo_pg_map = {}
        derived_map = {}

        # Policy groups should be one to one
        for agent_id in agent_id_class_map.keys():
            agent_class = self.agent_id_class_map[agent_id]
            self.policy_map[agent_class] = {}
            self.pseudo_policy_map[agent_class] = {}
            pg_map[agent_id], pseudo_pg_map[agent_id] = self.build_pg_dict(self.hierarchy.root_name,
                                                                           agent_class_observation_domains,
                                                                           agent_class_action_domains, agent_id,
                                                                           agent_class, pg_dict={}, pseudo_pg_dict={})

            if hasattr(self, 'derived_samples_hierarchy'):
                self.derived_policy_map = {}
                self.derived_policy_map[agent_class] = {}
                derived_map[agent_id] = self.build_derived_pg_map(self.derived_samples_hierarchy.root_name,
                                                                           agent_class_observation_domains,
                                                                           agent_class_action_domains, agent_id,
                                                                           agent_class, pg_dict={}, pseudo_pg_dict={})

                self.derived_policy_map[agent_id] = derived_map[agent_id]

            self.agent_policy_groups_map[agent_id] = agent_class
            self.hierarchical_policy_dict[agent_id] = pg_map[agent_id]

            self.completion_function_pg[agent_id] = pg_map[agent_id]
            self.pseudo_reward_cf_pg[agent_id] = pseudo_pg_map[agent_id]

            # Appends the head of the hierarchy's policy to the policy group
            policy_groups.append(pg_map[agent_class][self.hierarchy.root_name])

            # Initialize the action stack at head of hierarchy
            self.action_stack_dict[agent_id] = [self.hierarchy.root_name]

            self.action_sequence_dict[agent_id][self.hierarchy.root_name] = []

        return policy_groups


    # Todo: avoid passing pg_dict as arg in recursion
    def build_pg_dict(self, current_node: str, agent_class_observation_domains: Dict[int, ObservationDomain],
                      agent_class_action_domains: Dict[int, ActionDomain], agent_id: int, agent_class: int, pg_dict={},
                      pseudo_pg_dict={}, primitive=None) -> Tuple[Dict[str, PolicyGroup], Dict[str, PolicyGroup]]:
        """
        Constructs a flat dictionary mapping hierarchical 'agents' to two policy groups, one completion fn and one
        pseudo-reward completion function

        :param current_node: Current agent/subagent to create dict for
        :param agent_class_observation_domains: Maps agent to Obvs domain
        :param agent_class_action_domains: Maps an agent to its Action Domain
        :param agent_id: Actual agent id number
        :param pg_dict: partially filled Policy Group Dictionary
        :return: dictionary mapping a sub-agent (using its (str) name) to a PolicyGroup
        """
        # Can define the state abs of a state variable independent of parent
        if not primitive or current_node in self.h_obs_domain:
            obs_domain = self.h_obs_domain[current_node]
        else:
            # Primitive actions get obs domain of parent
            obs_domain = self.h_obs_domain[primitive]
        action_domain = self.hierarchy.domain_for_action(current_node)

        # Checks if policy already exists for sharing
        if current_node in self.policy_map[agent_class]:
            pseudo_policy = self.pseudo_policy_map[agent_class][current_node]
            policy = self.policy_map[agent_class][current_node]
        else:
            policy = self.policy_class(obs_domain, action_domain, self.top_config)
            pseudo_policy = copy.deepcopy(policy)
            self.pseudo_policy_map[agent_class][current_node] = pseudo_policy
            self.policy_map[agent_class][current_node] = policy

        model = self.algorithm_class.make_model(policy, self.top_config)
        pseudo_model = self.algorithm_class.make_model(pseudo_policy, self.top_config)

        pg_dict[current_node] = PolicyGroup(current_node, [agent_id], policy, model, self.max_time)
        pseudo_pg_dict[current_node] = PolicyGroup(current_node, [agent_id], pseudo_policy, pseudo_model, self.max_time)

        # Recurse on children
        for child in self.get_action_children(current_node):
            if child not in pg_dict:
                if self.is_primitive(child):
                    primitive = current_node
                else:
                    primitive = None
                pg_dict, pseudo_pg_dict = self.build_pg_dict(child, agent_class_observation_domains,
                                                             agent_class_action_domains,agent_id, agent_class,
                                                             pg_dict=pg_dict, pseudo_pg_dict=pseudo_pg_dict,
                                                             primitive=primitive)
        return pg_dict, pseudo_pg_dict

    def build_action_dict(self, current_action: str, action_dict: Dict[str, List] = {}) -> Dict[str, List]:
        """
        Maps each node (action/sub-agent represented by a string) to a list of children (str)

        :param current_action: Node in the graph
        :param action_dict: partially filled action dictionary
        :return: fully filled action dictionary for a single agent
        """
        children = self.hierarchy.action_hierarchy.get_children(current_action)
        action_dict[current_action] = children

        # Recurse to add children to the dictionary
        for child in children:
            self.build_action_dict(child, action_dict)

        return action_dict

    # This method uses the current node from the global stack for each agent
    def get_actions(self, observations: Dict[int, np.ndarray], use_max=False) -> Dict[int, np.ndarray]:
        """
        Implementation of the MAXQ action selection
        Returns a primitive action for each agent using an observation. Start at current non-primitive action in the
        agent action stack, traverses down hierarchy through non-primitives, storing each in a stack.
        Will only call non-terminated actions (for ex, cannot call Put without passenger)

        :param observations: Current Observation from Environment
        :param use_max: if true, agent use a greedy policy
        :return: dict mapping Agent to list of primitive action
        """
        all_actions = {}

        # if not use_max:
        #     self.append_means()

        for agent_id, observation in observations.items():
            self.clear_terminated_actions(agent_id)
            self.child_terminated_dict[agent_id] = {}
            current_node = self.action_stack_dict[agent_id][-1]
            abstracted_observation = observation

            # recurse on hierarchy until primitive action
            while not self.is_primitive(current_node):
                policy_group = self.completion_function_pg[agent_id][current_node]
                abstracted_observation = self.slice_observation(policy_group, observation, agent_id)

                # Get non-teminated action
                current_node, chosen_act = self.get_non_term_action(observation, current_node, agent_id,
                                                                    use_max=use_max)

                # For storing trajectory
                self._update_oar(agent_id, policy_group.pg_id, abstracted_observation, chosen_act, 0)

                # initialize sequence
                self.action_sequence_dict[agent_id][current_node] = []
                self.action_stack_dict[agent_id].append(current_node)

            # Store primitive action for state to execute
            primitive_action = np.asarray([self.primitive_action_map[current_node]])
            all_actions[agent_id] = primitive_action

            # append state to primitive action
            self.action_sequence_dict[agent_id][current_node].append(observation)

            # todo: may need to do something else here for sv

            # Each primitive has own value function so it must be [0]
            self._update_oar(agent_id, current_node, abstracted_observation, [0], 0)

        return all_actions

    def get_non_term_action(self, observation, parent, agent_id, use_max=False, q_val=False):
        """
        Selects an action from the children of parent
        Checks to make sure the action is not already terminated (i.e navigate_0_0 in 0,0)

        :param observation: Current state (non-abstracted)
        :param parent: Current node in hierarchy
        :param agent_id: Agent id
        :param use_max: Boolean dictating use of greedy policy
        :param q_val: if this value is being used for a q_val update, check if all children terminated
        :return: name of action, index of action
        """
        # Holds set of terminated children to avoid selecting them
        term_child = set()
        current_node = parent

        policy_group = self.completion_function_pg[agent_id][current_node]
        action_values = list(policy_group.policy.domain_act.get_action_range()[0])

        for action in action_values:
            current_node = self.int_to_action(action, parent)
            is_terminated = self.is_terminated(current_node, observation, agent_id)
            if not self.is_primitive(current_node) and is_terminated:
                term_child.add(action)

        if q_val:
            if len(term_child) == len(self.get_action_children(parent)):
                term_child = None

        chosen_act, _ = self.sample_actions(observation, parent, agent_id, use_max=use_max,
                                            term_child=term_child, use_pseudo=True)
        current_node = self.int_to_action(chosen_act[0], parent) if chosen_act != -1 else 'Terminate'
        return current_node, chosen_act

    def check_termination(self, observation: np.ndarray, agent_id):
        """
        Checks current subtasks for termination and adds to child_terminated_dict

        :param observation: current state (unabstracted)
        :param agent_id: actual agent id
        """
        action_stack = self.action_stack_dict[agent_id]
        parent = action_stack[0]
        add_all = False

        for current_node in action_stack:
            if self.is_terminated(current_node, observation, agent_id) or add_all:
                add_all = True
                if not current_node == action_stack[0]:
                    self.child_terminated_dict[agent_id][parent] = current_node
            parent = current_node

        self.prepare_terminated(agent_id, observation)

    def prepare_terminated(self, agent_id, observation):
        """
        Iterates through all nodes with terminated children and prepares for updating
        Works backwards up stack to cascade reward
        :param agent_id: Agent Id to update
        :param observation: Observation of agent to update
        :return:
        """
        action_stack = self.action_stack_dict[agent_id][:-1]
        for action in reversed(action_stack):
            if action in self.child_terminated_dict[agent_id]:
                child = self.child_terminated_dict[agent_id][action]
                self.prepare_child(child, action, agent_id, observation)

    def prepare_child(self, child_action, parent, agent_id, observation):
        """
        Prepares node with terminated children for updating
        Appends to trajectory, checks pseudo-reward

        :param child_action: terminated child
        :param parent: parent of terminated child
        :param agent_id: agent id to update
        :param observation: state for updating
        :return:
        """
        pseudo_r = {}
        completion_fn = self.completion_function_pg[agent_id][parent]
        pseudo_completion_fn = self.pseudo_reward_cf_pg[agent_id][parent]

        # Get reward for cascading reward
        reward_to_add = self.current_reward_dict[agent_id][child_action]
        self.current_reward_dict[agent_id][completion_fn.pg_id] += reward_to_add
        pg_obs, pg_act, pg_rew = self.translate_pg_signal(self.abstracted_obs_dict[agent_id],
                                                          self.current_action_dict[agent_id],
                                                          self.current_reward_dict[agent_id])

        self.child_terminated_dict[agent_id][parent] = child_action
        completion_fn.append(pg_obs, pg_act, pg_rew)

        pseudo_r[parent] = self.check_pseudo_reward(child_action, observation, agent_id)
        pseudo_completion_fn.append(pg_obs, pg_act, pseudo_r)

    def update_terminated(self, observations: Dict[int, np.ndarray], agent_id, traj_done=False):
        """
        Updates completion functions of subtasks with terminated children

        :param observations: observation (state)
        :param agent_id: actual agent id
        """
        action_stack = self.action_stack_dict[agent_id]

        # traverse up action_stack (for sequence propigation)
        for current_node in reversed(action_stack):
            observation = observations[agent_id]

            if current_node in self.child_terminated_dict[agent_id].keys():
                child_action = self.child_terminated_dict[agent_id][current_node]
                completion_fn = self.completion_function_pg[agent_id][current_node]
                pseudo_cf = self.pseudo_reward_cf_pg[agent_id][current_node]
                child_sequence = self.action_sequence_dict[agent_id][child_action]
                abstracted_observation = self.slice_observation(completion_fn, observation, agent_id)
                next_act, max_action_index = self.get_non_term_action(observation, current_node, agent_id, use_max=True,
                                                                      q_val=True)

                # Todo return action and value at once?
                v_s_prime = self.eval_max_node(observation, next_act, agent_id)

                # If parent of current node is also terminated, completion function is 0
                # This is last action parent will choose before termination
                if ((current_node in self.child_terminated_dict[agent_id].values())
                        or (current_node == self.hierarchy.root_name and traj_done)):
                    cf_value = 0
                    pseudo_cf_value = 0

                else:
                    cf_value = completion_fn.policy.eval(abstracted_observation)[max_action_index[0]]
                    pseudo_cf_value = pseudo_cf.policy.eval(abstracted_observation)[max_action_index[0]]

                n = 1
                for state in child_sequence:
                    child_observation = self.slice_observation(completion_fn, state, agent_id)
                    # Update both completion functions
                    self.algorithm.completion_function_update(completion_fn, pseudo_cf, v_s_prime, cf_value,
                                                              pseudo_cf_value, n, child_observation)

                    # Append observation onto the front of current sequence
                    self.action_sequence_dict[agent_id][current_node].insert(n - 1, state)
                    n = n + 1

    def make_additional_var(self, termination_pred, current_node, agent_id, full_observation):
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
                additional_var['target'] = self.hierarchy.action_hierarchy.ground_var_map[current_node]['target']

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

    def sample_actions(self, observation: np.ndarray, current_node: str, agent_id: int, use_max=False, use_pseudo=False,
                       term_child=None):
        """
        A hierarchical action selection outlined in MaxQ. Finds the action values
        using the hierarchy then uses the action sampler to select action

        Modeled after algorithm on page 25 in MaxQ
        :param observation: state
        :param current_node: current action in hierarchy
        :param agent_id: actual agent id
        :param use_max: whether to use greedy action
        :return: ndarray with action index(s)
        """

        if use_pseudo:
            policy_group = self.pseudo_reward_cf_pg[agent_id][current_node]
        else:
            policy_group = self.completion_function_pg[agent_id][current_node]

        action_values = self.eval_children(observation, current_node, agent_id)

        # Used to track terminated actions index to map back to actual indexs
        index_list = np.arange(len(action_values))

        # Removes actions that are terminated in the current state
        if term_child:
            action_values = np.delete(action_values, [c for c in term_child])
            index_list = np.delete(index_list, [c for c in term_child])

        if len(action_values) == 1:
            actions = [0]
        elif len(action_values) == 0:
            raise ValueError('No Actions Left')
        elif use_max:
            # Return argmax action
            _, sampler_values = policy_group.policy.sampler.sample(action_values)
            vals = sampler_values[agent_id]
            max_action = max(vals.items(), key=operator.itemgetter(1))[0]
            actions = [max_action]
        else:
            actions, sv = policy_group.policy.sampler.sample(action_values)

        action_val = action_values[actions[0]]

        # Remap action back to 'actual' index (after deleting terminated indexes)
        actions = [index_list[actions[0]]]
        return actions, action_val

    def eval_children(self, states: np.ndarray, current_node, agent_id):
        """
        Finds values of children of a node for action selection, uses eval_max_node to get children values

        :param states: current state
        :param current_node: node/location in hierarchy
        :param agent_id: id of actual agent
        :return: array with the values of a nodes children
        """
        child_values = []
        pcf = self.pseudo_reward_cf_pg[agent_id][current_node]
        abs_obs = self.slice_observation(pcf, states, agent_id)
        pseudo_val = pcf.policy.eval(abs_obs)
        for ind, child in enumerate(self.hierarchy.action_hierarchy.get_children(current_node)):
            cf_val = pseudo_val[ind]
            child_val = self.eval_max_node(states, child, agent_id, use_pseudo=False)
            child_values.append(cf_val + child_val)

        return np.asarray(child_values)

    # TODO should we always be using pseudo rew (p32)
    def eval_max_node(self, observation: np.ndarray, current_node, agent_id, use_pseudo=False):
        """
        Returns the value of a node in the hierarchy
        Evaluate Max Node algorithm from page 26 of maxQ, except it returns value of current node not next action

        :param observation: Current state (un-abstracted)
        :param current_node: Current node in hierarchy to get value of
        :param agent_id: Agent Id
        :return: value of a node
        """
        if use_pseudo:
            policy_group = self.pseudo_reward_cf_pg[agent_id][current_node]
        else:
            policy_group = self.completion_function_pg[agent_id][current_node]

        abstracted_observation = self.slice_observation(policy_group, observation, agent_id)

        # primitive action, get value V(s,a)
        if self.is_primitive(current_node):
            return policy_group.policy.eval(abstracted_observation)[0]

        #  V(i,s) = max(q(i,s,a)) where Q(i,s,a) is the C(a,s) + V(a,s)
        else:
            children = self.get_action_children(current_node)
            max_child = float("-inf")
            max_act = None
            c_values = policy_group.policy.eval(abstracted_observation)
            for ind, child in enumerate(children):
                child_value = self.eval_max_node(observation, child, agent_id)
                completion_function_value = c_values[ind]
                value = child_value + completion_function_value

                if value > max_child:
                    max_child = value
                    max_cv = child_value
        return max_child

    def append_pg_signals(self, a_obs: Dict[int, np.ndarray], a_act: Dict[int, np.ndarray],
                          a_rew: Dict[int, Union[int, np.ndarray]], done: bool):
        """
        Appends only to primitive action

        :param a_obs: Agent-organized observation map
        :param a_act: Agent-organized action map
        :param a_rew: Agent-organized reward map
        :param done: If the observed state is terminal
        """
        # For each agent, check the termination status of their action stack in the current state
        for agent_id in self.agent_ids:
            primitive_action = self.action_stack_dict[agent_id][-1]
            policy_group = self.hierarchical_policy_dict[agent_id][primitive_action]

            completion_fn = self.completion_function_pg[agent_id][primitive_action]
            pseudo_completion_fn = self.pseudo_reward_cf_pg[agent_id][primitive_action]

            self.current_reward_dict[agent_id][policy_group.pg_id] = a_rew[agent_id]

            pg_obs, pg_act, pg_rew = self.translate_pg_signal(self.abstracted_obs_dict[agent_id],
                                                              self.current_action_dict[agent_id],
                                                              self.current_reward_dict[agent_id])
            completion_fn.append(pg_obs, pg_act, pg_rew)
            pseudo_completion_fn.append(pg_obs, pg_act, pg_rew)

    def translate_pg_signal(self, a_observations: Dict[int, np.ndarray], a_actions: Dict[int, np.ndarray],
                            a_rewards: Dict[int, Union[int, np.ndarray]]):
        return a_observations, a_actions, a_rewards

    def is_primitive(self, current_node) -> bool:
        """
        Is current node primitive?

        :param current_node: node to check if primitive
        :return: True if current node is primtive
        """
        if not self.hierarchy.action_hierarchy.get_children(current_node):
            return True
        else:
            return False

    def learn_update(self):
        """
        Updates value function for primitive actions
        """
        for agent_id in self.agent_ids:
            # get primitive action
            term_node = self.action_stack_dict[agent_id][-1]
            completion_fn = self.completion_function_pg[agent_id][term_node]
            pcf = self.pseudo_reward_cf_pg[agent_id][term_node]

            # update primitive action value function, since this does not need the next state
            self.algorithm.primitive_update(completion_fn.policy, completion_fn.model, completion_fn.trajectory)
            self.algorithm.primitive_update(pcf.policy, pcf.model, pcf.trajectory)

    def check_all_agent_termination(self, observations: np.array):
        """
        Checks all actions of all agents for termination

        :param observations: All states, mapped by agent id
        """
        for agent_id, observation in observations.items():
            self.check_termination(observation, agent_id)

    def hierarchical_update(self, observations: np.array, traj_done = False):
        """
        Updates nodes with terminated children within hierarchy

        :param observations: all states mapped by agent id
        :param traj_done: is the parent action terminated
        """
        for agent_id, observation in observations.items():
            self.update_terminated(observations, agent_id, traj_done)

    def is_terminated(self, current_node: str, observation: np.array, agent_id: int):
        """
        Returns True if an subtask is terminated

        :param current_node: current subtask
        :param observation: current state
        :param agent_id: agent id
        :return: True if current node is terminated
        """
        # Primitive action always terminates
        if self.is_primitive(current_node):
            return True
        else:
            termination_pred = self.hierarchy.action_hierarchy.get_termination_predicate(current_node)
            cf = self.completion_function_pg[agent_id][current_node]
            obs_domain = cf.policy.domain_obs
            obs = self.slice_observation(cf, observation, agent_id)
            additional_vars = self.make_additional_var(termination_pred, current_node, agent_id, observation)
            terminated = predicates.evaluate_predicate(termination_pred, obs_domain, obs, additional_vars)
            return terminated

    def check_pseudo_reward(self, current_node: str, observation: np.array, agent_id: int):
        """
        Returns pseudo reward of terminated actions

        :param current_node: Current node in hierarchy
        :param observation: Current state
        :param agent_id: Agent id
        :return: pseudo reward
        """
        if current_node not in self.child_terminated_dict[agent_id].values():
            raise ValueError("Pseudo Reward only for terminated actions")
        # No Pseudo R for primitives
        elif self.is_primitive(current_node):
            return 0
        else:
            pseudo_r = self.hierarchy.action_hierarchy.get_pseudo_rewards(current_node)
            obs_domain = self.completion_function_pg[agent_id][current_node].policy.domain_obs
            # Is there any pseudo R?
            if pseudo_r[0]:
                for predicate in pseudo_r:
                    additional_vars = self.make_additional_var([predicate[0]], current_node, agent_id)
                    if predicates.evaluate_predicate([predicate[0]], obs_domain, observation, additional_vars):
                        return predicate[-1]
            return 0

    def clear_terminated_actions(self, agent_id):
        """
        Clears terminated actions from action stack

        :param agent_id: ID of agent to clear stack
        """
        for ind, action in enumerate(self.action_stack_dict[agent_id]):
            if action in self.child_terminated_dict[agent_id].values():
                self.current_reward_dict[agent_id].pop(action)
                self.current_action_dict[agent_id].pop(action)
                self.abstracted_obs_dict[agent_id].pop(action)
                self.action_sequence_dict[agent_id].pop(action)
                del (self.action_stack_dict[agent_id][ind:])

    def slice_observation(self, pg: PolicyGroup, observation: np.ndarray, agent_id: int):
        """
        Abstracts an observation according to the policy groups observation domain

        :param pg: Policy Group
        :param observation: Full Observation
        :param agent_id: ID of Actual Agent
        :return: abstracted observation
        """
        sliced_obs = []
        agent_class = self.agent_id_class_map[agent_id]
        for item in pg.policy.domain_obs.items:
            domain_slice = self.agent_slice_dict[agent_class][item.name]
            sliced_obs[pg.policy.domain_obs.index_for_name(item.name)] = observation[domain_slice]
        return np.asarray(sliced_obs)

    def all_agent_slice_dict(self, agent_class_observation_domains: Dict[int, ObservationDomain]):
        """
        Maps each agent to a dictionary mapping each state variable in observation domain to its domain slicer

        :param agent_class_observation_domains: Maps actual agent to Obs Domain
        :return: dictionary mapping agents to abstracted observation domain
        """
        agents_slice_dict = {}
        for agent in agent_class_observation_domains.keys():
            agents_slice_dict[agent] = self.make_slice_dict(agent_class_observation_domains
                                                                [self.agent_id_class_map[agent]])
        return agents_slice_dict

    def make_slice_dict(self, obs_domain: ObservationDomain):
        """
        Helper for all_agent_slice_dict

        :param obs_domain: Observation Domians
        :return: dictionary mapping each state var to its domain slice
        """
        slice_dict = {}
        for item in obs_domain.items:
            slice_dict[item.name] = obs_domain.index_for_name(item.name)
        return slice_dict

    def action_to_int(self, child, parent) -> int:
        """
        Convert (string) action to its corresponding action

        :param child: child action (string)
        :param parent: parent action (string)
        :return: index of action
        """
        children = self.get_action_children(parent)
        return children.index(child)

    def int_to_action(self, child, parent) -> str:
        """
        Convert index of an action to a string

        :param child: child action (string)
        :param parent: parent action (string)
        :return: string name of action
        """
        children = self.get_action_children(parent)
        return children[child]

    def get_action_children(self, current_node):
        """
        Returns children of current node
        :param current_node:
        :return: the children of current_node
        """
        return self.hierarchy.action_hierarchy.get_children(current_node)

    def get_parent(self, current_node, agent_id):
        for ind, act in enumerate(self.action_stack_dict[agent_id]):
            if current_node == act:
                return self.action_stack_dict[agent_id][ind - 1]
        raise ValueError('No parent of requested action')

    def _update_oar(self, agent_id, pg_id, obs, act, rew):
        """
        Update current dictionaries with observation, action, reward
        """

        self.abstracted_obs_dict[agent_id][pg_id] = obs
        self.current_action_dict[agent_id][pg_id] = act
        self.current_reward_dict[agent_id][pg_id] = rew

    def _init_current_dicts(self, agent_list: List):
        """
        Initialize all dictionaries

        :param agent_list: List of agents
        :return:
        """
        for agent in agent_list:
            self.abstracted_obs_dict[agent] = {}
            self.current_action_dict[agent] = {}
            self.current_reward_dict[agent] = {}
            self.action_sequence_dict[agent] = {}
            self.child_terminated_dict[agent] = {}

    # Copied from independent system
    def transfer_domain(self, domain_transfer_message: DomainTransferMessage) -> AgentSystem:
        # Only supports deletion (thus far)
        # never ever ever should there be a real runtime error due to bad config
        assert domain_transfer_message.add_agent_class_id_map is None

        # TODO consider refactoring common code in future?
        mapped_ids = domain_transfer_message.remap_id_list(self.agent_ids)

        # All deletions performed. Only remaps left.
        # Remap IDs (again) according to rule [remapped_ids] -> range(len(remapped_ids))
        new_agent_ids = []
        new_agent_id_class_map = {}
        new_agent_class_id_map = {}
        new_agent_policy_groups_map = {}
        new_policy_groups = []
        for canonical_id, id_from in enumerate(mapped_ids):
            new_agent_ids.append(canonical_id)
            agent_class = self.agent_id_class_map[id_from]
            new_agent_id_class_map[canonical_id] = agent_class
            if agent_class not in new_agent_class_id_map:
                new_agent_class_id_map[agent_class] = []
            new_agent_class_id_map[agent_class].append(canonical_id)
            pg = self.agent_policy_groups_map[id_from][0]
            new_policy_groups.append(pg)
            pg.agents = [canonical_id]
            new_agent_policy_groups_map[canonical_id] = [pg]

        self.agent_ids = new_agent_ids
        self.agent_class_id_map = new_agent_class_id_map
        self.agent_id_class_map = new_agent_id_class_map
        self.agent_policy_groups_map = new_agent_policy_groups_map
        self.policy_groups = new_policy_groups

        return self

    def reset(self):
        """
        Reset episode-tied information when starting a new episode.
        """

        for pg in self.policy_groups:
            pg.allocate_new_trajectory()

        for agent_id in self.agent_ids:
            for node in self.completion_function_pg[agent_id]:
                self.completion_function_pg[agent_id][node].allocate_new_trajectory()
                self.pseudo_reward_cf_pg[agent_id][node].allocate_new_trajectory()

        for agent_id in self.agent_ids:
            # Initialize the action stack at head of hierarchy
            self.action_stack_dict[agent_id] = [self.hierarchy.root_name]
            self.action_sequence_dict[agent_id][self.hierarchy.root_name] = []
            self.clear_terminated_actions(agent_id)
            self.child_terminated_dict[agent_id] = {}

    def append_means(self):
        pg_dict = self.completion_function_pg[0]
        nav_means = []
        primitive_means = []
        root_mean = 0.0
        get_put_means = []
        putdown_means = []
        pickup_means = []
        for pg in pg_dict:
            if "Navigate" in pg:
                nav_means.append(np.mean(pg_dict[pg].policy.table))
            elif pg in {'East', 'West', 'North', 'South', 'East1', 'West1', 'North1', 'South1'}:
                primitive_means.append(np.mean(pg_dict[pg].policy.table))
            elif pg == 'Root':
                root_mean = np.mean(pg_dict[pg].policy.table)
            elif pg in {'Get', 'Put'}:
                get_put_means.append(np.mean(pg_dict[pg].policy.table))
            elif pg == 'Pickup':
                pickup_means.append(np.mean(pg_dict[pg].policy.table))
            elif pg in 'Putdown':
                putdown_means.append(np.mean(pg_dict[pg].policy.table))

        if nav_means:
            self.navigate_means.append(round(np.mean(nav_means), 3))

        self.primitive_means.append(round(np.mean(primitive_means), 3))
        if get_put_means:
            self.get_put_means.append(round(np.mean(get_put_means), 3))

        self.pickup_means.append(round(np.mean(pickup_means), 3))
        self.putdown_means.append(round(np.mean(putdown_means), 3))
        self.root_mean.append(round(np.mean(root_mean), 3))

    def end_episode(self, learn: bool = False):
        """
        Prepares hierarchy for next learning episode

        :param learn:
        :return:
        """

        for agent_id in self.agent_ids:
            for pg in self.completion_function_pg[agent_id].keys():
                if hasattr(self.completion_function_pg[agent_id][pg].policy, 'sampler') and learn:
                    self.completion_function_pg[agent_id][pg].policy.sampler.update_params()
                self.completion_function_pg[agent_id][pg].trajectory.cull()

                if hasattr(self.completion_function_pg[agent_id][pg].policy, 'sampler') and learn:
                    self.pseudo_reward_cf_pg[agent_id][pg].policy.sampler.update_params()
                self.pseudo_reward_cf_pg[agent_id][pg].trajectory.cull()


