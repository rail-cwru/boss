from typing import Dict, List, Tuple, Union, Any

import numpy as np
import copy as copy
from common.trajectory import Trajectory
from domain.ActionDomain import ActionDomain
from domain.hierarchical_domain import HierarchicalActionDomain
from domain.observation import ObservationDomain
from agentsystem import AgentSystem
from agentsystem.util import PolicyGroup
from config import Config, AbstractModuleFrame, ConfigItemDesc
from common.domain_transfer import DomainTransferMessage
from common.properties import Properties
from policy import Policy
from common.predicates import predicates
from algorithm import Algorithm
import operator
from agentsystem.HierarchicalSystem import HierarchicalSystem
from sampler.PolledSampler import PolledSampler

class WeaklyPolledSampler(PolledSampler, AbstractModuleFrame):

    """
    An agent system designed for random hierarchical policy sampling for offline learning
    """

    def sample_actions(self, observation: np.ndarray, current_node: str, agent_id: int, use_max=False,
                       use_pseudo=False, term_child=None):
        """
        Selects a random action, ignoring terminated actions in term_child

        Modeled after algorithm on page 25 in MaxQ
        :param observation: state
        :param current_node: current action in hierarchy
        :param agent_id: actual agent id
        :param use_max: whether to use greedy action
        :return: ndarray with action index(s)
        """

        raise NotImplementedError("Not completely implemented")
        policy_group = self.completion_function_pg[agent_id][current_node]
        action_values = list(policy_group.policy.domain_act.get_action_range()[0])

        # Used to track terminated actions index to map back to actual indexs
        index_list = np.arange(len(action_values))

        # Removes actions that are terminated in the current state
        if term_child:
            for child in term_child:
                action_values = np.delete(action_values, child)
                index_list = np.delete(index_list, child)

        if len(action_values) == 0:
            raise ValueError('No Actions Left')
        elif len(action_values) == 1:
            actions = [index_list[0]]
        else:
            exists_nonterm_sibling = self.check_term_siblings(current_node, observation, agent_id)
            #print(current_node, exists_nonterm_sibling)
            if exists_nonterm_sibling:
                index_list = np.append(index_list, -1)

            actions = [np.random.choice(index_list)]

            if actions == [-1]:
                return -1, -1

        action_val = actions[0]

        # Remap action back to 'actual' index (after deleting terminated indexes)
        return actions, action_val

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
        raise NotImplementedError

        for agent_id, observation in observations.items():
            self.add_s_prime(observation, agent_id)

            self.clear_terminated_actions(agent_id)
            self.child_terminated_dict[agent_id] = {}
            current_node = self.action_stack_dict[agent_id][-1]
            abstracted_observation = observation
            self.term_subtasks[agent_id] = []

            # recurse on hierarchy until primitive action
            while not self.is_primitive(current_node):
                policy_group = self.completion_function_pg[agent_id][current_node]
                abstracted_observation = self.slice_observation(policy_group, observation, agent_id)

                # Get non-teminated action
                current_node, chosen_act = self.get_non_term_action(observation, current_node, agent_id,
                                                                    use_max=use_max)
                if chosen_act == -1 or current_node == 'Terminate':
                    # remove last node (since it is being terminated)
                    #print(self.action_stack_dict[0])
                    self.action_stack_dict[0].pop(-1)
                    current_node = self.action_stack_dict[0][-1]
                    continue

                # For storing trajectory
                self._update_oar(agent_id, policy_group.pg_id, abstracted_observation, chosen_act, 0)

                # initialize sequence
                self.action_sequence_dict[agent_id][current_node] = []
                self.action_stack_dict[agent_id].append(current_node)

            # Store primitive action for state to execute
            primitive_action = np.asarray([self.primitive_action_map[current_node]])
            all_actions[agent_id] = primitive_action

            if self.collect_inhibited or self.collect_abstract:
                self.check_all_subtask_termination(observation, agent_id)

            # TODO: Confirm that this is done correctly
            if self.collect_inhibited:
                self.generate_inhibited_samples(observation, agent_id)
            if self.collect_abstract:
                self.generate_abstract_samples(observation, policy_group, agent_id, primitive_action)

            # append state to primitive action
            self.action_sequence_dict[agent_id][current_node].append(observation)

            # Each primitive has own value function so it must be [0]
            self._update_oar(agent_id, current_node, abstracted_observation, [0], 0)

        return all_actions

    def check_term_siblings(self, current_node, observation, agent_id):
        """
        Returns True if current_node has non-terminated sibling
        Returns False if current_node is the only non-terminated child of its parent
        :param current_node:
        :return:
        """

        if current_node == "Root":
            return False

        action_stack = self.action_stack_dict[agent_id]
        parent = -1
        for ind, i in enumerate(action_stack):
            if i == current_node:
                parent = action_stack[ind-1]

        if parent == -1:
            raise ValueError('Parent Not Found')
        else:
            siblings = self.get_action_children(parent)
            for sibling in siblings:
                if (not self.is_terminated(sibling, observation, agent_id) and not sibling == current_node) or self.is_primitive(sibling):
                    return True
        return False

    def check_all_subtask_termination(self, observation, agent_id):
        current_node = "Root"
        self._loop_children(current_node, observation, agent_id)

    def _loop_children(self, current_node, observation, agent_id):
         for child in self.get_action_children(current_node):
             if not self.is_primitive(child):
                 if self.is_terminated(child, observation, agent_id):
                     self.term_subtasks[agent_id].append(child)
                 else:
                     self._loop_children(child, observation, agent_id)

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
                self.term_subtasks[agent_id].append(current_node)

        if q_val:
            if len(term_child) == len(self.get_action_children(parent)):
                term_child = None

        chosen_act, _ = self.sample_actions(observation, parent, agent_id, use_max=use_max,
                                            term_child=term_child, use_pseudo=True)
        current_node = self.int_to_action(chosen_act[0], parent) if chosen_act != -1 else 'Terminate'
        return current_node, chosen_act

