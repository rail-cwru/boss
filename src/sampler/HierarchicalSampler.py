import numpy as np
from config import AbstractModuleFrame
from agentsystem.HierarchicalSystem import HierarchicalSystem

class HierarchicalSampler(HierarchicalSystem, AbstractModuleFrame):
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
        else:
            actions = [np.random.choice(index_list)]

        action_val = actions[0]

        # Remap action back to 'actual' index (after deleting terminated indexes)
        return actions, action_val