from typing import Dict, List, Tuple, Union, Any

import numpy as np
from domain.ActionDomain import ActionDomain
from domain.observation import ObservationDomain
from config import AbstractModuleFrame
from config import Config
from sampler.PolledSampler import PolledSampler

class HierarchicalUniform(PolledSampler, AbstractModuleFrame):
    """
    HierarchicalUniform (HUF) sampler
    Selects samples using a uniform random policy over reachable actions
    while deriving samples from the original hierarchy

    The HUF sampler takes no additional parameters than the Polled sampler

    @author Eric Miller
    @contact edm54@case.edu
    """

    def __init__(self, agent_id_class_map: Dict[int, int], agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain], auxiliary_data: Dict[str, Any],
                 config: Config):
        super(HierarchicalUniform, self).__init__(agent_id_class_map, agent_class_action_domains,
                                                            agent_class_observation_domains, auxiliary_data, config)

    # This method uses the current node from the global stack for each agent
    def get_actions(self, observations: Dict[int, np.ndarray], use_max=False, add_s_prime=True) -> Dict[int, np.ndarray]:
        """
        This action selection motivates the agent to explore unexplored actions,
        with the ability to use the abstracted state space and can generate abstracted samples
        """

        all_actions = {}
        for agent_id, observation in observations.items():
            if not self.first_action and add_s_prime:
                self.add_s_prime(observation, agent_id)

            pg = self.agent_policy_groups_map[agent_id][0]
            self.term_subtasks[agent_id] = []

            if self.optimistic:
                current_node, action = self.get_optimistic_flat_samples(observation, "Root", agent_id)
            else:
                # Get non-terminated action tat has been visited least
                current_node, action = self.get_flat_samples(observation, "Root", agent_id)

            all_actions[agent_id] = action

            abstracted_observation = self.slice_observation(pg, observation, agent_id)
            self._update_oar(agent_id, pg.pg_id, abstracted_observation, action, 0)
            self._collect_derived(observation, self.completion_function_pg[agent_id][current_node], action, agent_id)

            if self.kl_div:
                if len(self.num_states) != len(observation):
                    obs = self._map_table_indices(observation, pg.policy)
                else:
                    obs = self.slice_observation(pg, observation, agent_id)

                index = self.state_action_basis.get_state_action_index(obs, action[0])
                self.state_action_visits[index] += 1

            self.first_action = False
        return all_actions

    def get_flat_samples(self, observation, current_node, agent_id):
        """
        Samples from all non-terminated primitive actions from the flat distribution
        :return: a non-terminated primitive action
        """
        abstracted_obs = self.abstract_state(observation, current_node, agent_id)

        state_index = self.state_action_basis.get_state_action_index(abstracted_obs, 0)

        if self.inhibited_actions_arr[state_index] is None:
            # Collect all primitives that are not terminated
            non_term_prim, reachable_subtasks = self.get_all_reachable_non_term_actions(observation, "Root",
                                                                                        agent_id)
            term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
            self.inhibited_actions_arr[state_index] = (term_prim, reachable_subtasks)
        else:
            [term_prim, _] = self.inhibited_actions_arr[state_index]
            non_term_prim = [i for i in self.primitive_action_map.keys() if i not in term_prim]

        current_node = [np.random.choice(non_term_prim)][0]
        chosen_act = [self.primitive_action_map[current_node]]

        index = self.state_action_basis.get_state_action_index(abstracted_obs, chosen_act[0])
        self.state_action_visits[index] += 1
        self.print_counts(abstracted_obs, non_term_prim)

        if self.collect_abstract:
            act_basis = self.state_action_basis_map[current_node]
            act_abstracted_obs = self.abstract_state(observation, current_node, agent_id)

            act_index = act_basis.get_state_action_index(act_abstracted_obs, 0)
            self.state_action_visits_map[current_node][act_index] += 1

        return current_node, chosen_act

    def get_optimistic_flat_samples(self, observation, current_node, agent_id):
        """
        Samples from all non-terminated primitive actions from the flat distribution
        :return: a non-terminated primitive action
        """

        abstracted_obs = self.abstract_state(observation, current_node, agent_id)

        state_index = self.state_action_basis.get_state_action_index(abstracted_obs, 0)

        if self.inhibited_actions_arr[state_index] is None:
            # Collect all primitives that are not terminated
            non_term_prim, reachable_subtasks = self.get_all_reachable_non_term_actions(observation, "Root",
                                                                                        agent_id)
            term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
            self.inhibited_actions_arr[state_index] = (term_prim, reachable_subtasks)
        else:
            [term_prim, _] = self.inhibited_actions_arr[state_index]
            non_term_prim = [i for i in self.primitive_action_map.keys() if i not in term_prim]

        if self.use_primitive_distribution:
            num_visits = []
            for act in non_term_prim:
                act_abstracted_obs = self.abstract_state(observation, act, agent_id)
                act_basis = self.state_action_basis_map[act]
                act_index = act_basis.get_state_action_index(act_abstracted_obs, 0)
                state_act_visits = self.state_action_visits_map[act][act_index]
                num_visits.append(state_act_visits)
            num_visits = np.asarray(num_visits)
        else:

            # Get the index for each non-terminated action, and number of visits for each action, A
            sa_index_list = [self.state_action_basis.get_state_action_index(abstracted_obs, self.primitive_action_map[action])
                            for action in non_term_prim]
            num_visits = np.asarray([self.state_action_visits[index] for index in sa_index_list])

        # Check if state is unvisited
        if num_visits.max() == 0:
            current_node = [np.random.choice(non_term_prim)][0]
            chosen_act = [self.primitive_action_map[current_node]]
        else:
            # Select min with random tiebreak
            chosen_act = [np.random.choice(np.where(num_visits == num_visits.min())[0])]
            current_node = non_term_prim[chosen_act[0]]

            # If there are terminated actions, need to shift the selected action to match the entire action space
            if len(non_term_prim) != self.num_actions:
                current_node = non_term_prim[chosen_act[0]]

            # Convert action name to numeric action
            chosen_act = [self.primitive_action_map[current_node]]

        index = self.state_action_basis.get_state_action_index(abstracted_obs, chosen_act[0])
        self.state_action_visits[index] += 1

        if self.collect_abstract:
            act_basis = self.state_action_basis_map[current_node]
            act_abstracted_obs = self.abstract_state(observation, current_node, agent_id)

            act_index = act_basis.get_state_action_index(act_abstracted_obs, 0)
            self.state_action_visits_map[current_node][act_index] += 1

        return current_node, chosen_act

    def print_counts(self, abstracted_obs, nt_prim):
        visit_map = {}
        for current_node in nt_prim:
            chosen_act = [self.primitive_action_map[current_node]]
            index = self.state_action_basis.get_state_action_index(abstracted_obs, chosen_act[0])
            visit_map[current_node] = self.state_action_visits[index]

    def get_sample_probability(self, sample, agent_id=0) -> float:

        state = self.map_obs(sample[0], agent_id)
        state_index = self.state_action_basis.get_state_action_index(state, 0)

        if self.inhibited_actions_arr[state_index] is None:
            # Collect all primitives that are not terminated
            non_term_prim, reachable_subtasks = self.get_all_reachable_non_term_actions(sample[0], "Root",
                                                                                        agent_id)
            term_prim = [i for i in self.primitive_action_map.keys() if i not in non_term_prim]
            self.inhibited_actions_arr[state_index] = (term_prim, reachable_subtasks)
        else:
            [term_prim, _] = self.inhibited_actions_arr[state_index]
            non_term_prim = [i for i in self.primitive_action_map.keys() if i not in term_prim]

        return 1/len(non_term_prim)
