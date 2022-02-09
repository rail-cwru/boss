from typing import Dict, List, Any

import numpy as np
from domain.ActionDomain import ActionDomain
from domain.observation import ObservationDomain
from config import AbstractModuleFrame, ConfigItemDesc
from agentsystem.HierarchicalSystem import HierarchicalSystem
from config import Config, checks
from policy.function_approximator.basis_function.ExactBasis import ExactBasis
from sampler import PolledSampler, HierarchicalUniform, FlattenedPolledSampler
import random

class BOSS(HierarchicalSystem, AbstractModuleFrame):
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
            ConfigItemDesc('samplers_list', checks.list, 'List of Samplers to rotate through',
                           default=["HUF", "Polled", "TDF_1", "BUF_1"], optional=True),
            ConfigItemDesc('ucb_coef', checks.positive_integer,
                           'UCB multiplier',
                           default=2.0, optional=True),
            ConfigItemDesc('check_dist',
                           checks.boolean,
                           'saves the distribution of samples per action',
                           default=False,
                           optional=True),
        ]

    def __init__(self, agent_id_class_map: Dict[int, int], agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain], auxiliary_data: Dict[str, Any],
                 config: Config, hierarchy_map: Dict):

        self.term_subtasks = {}
        super(BOSS, self).__init__(agent_id_class_map, agent_class_action_domains,
                                   agent_class_observation_domains, auxiliary_data, config)

        self.min_reward = config.sampler.min_reward if hasattr(config.sampler, 'min_reward') else -100
        self.save_target = config.sampler.save_target if hasattr(config.sampler, 'save_target') else False
        self.check_dist = config.sampler.check_dist if hasattr(config.sampler, 'check_dist') else False

        self.save_name = config.sampler.save_name if hasattr(config.sampler, 'save_name') else 'chair_target.list'

        self.collect_inhibited = config.sampler.collect_inhibited
        self.collect_abstract = config.sampler.collect_abstract
        self.use_primitive_distribution = config.sampler.use_primitive_dist if hasattr(config.sampler, 'use_primitive_dist') else False

        # Shuffle order of samplers list
        old_state = random.getstate()
        random.seed()
        # Randomly reorder samplers list
        random.shuffle(config.sampler.samplers_list)
        random.setstate(old_state)

        self.samplers_list = config.sampler.samplers_list

        self.ucb_coef = config.sampler.ucb_coef

        self.optimistic = config.sampler.optimistic if hasattr(config.sampler, 'optimistic') else False
        print("Use balanced wandering:", self.optimistic)

        if self.save_target:
            print("Save Target")

        self.inhibited_abstract_samples = []

        self._init_derived()
        self.per_sample_derived = []
        self.current_ep_derived = []
        self.first_action = True
        self.num_states = [domain_item.num_values() for domain_item in
                           agent_class_observation_domains[agent_id_class_map[0]].items]

        self.num_actions = agent_class_action_domains[0].full_range
        self.temperature = config.sampler.temperature if hasattr(config.sampler, "temperature") else 1.0
        self._create_state_visits()

        self.sampler_object_dict = {}
        self.hierarchy_map = hierarchy_map

        if "Polled" in self.samplers_list:
            self.polled_sampler = PolledSampler.PolledSampler(agent_id_class_map,
                                                              agent_class_action_domains,
                                                              agent_class_observation_domains,
                                                              auxiliary_data,

                                                              config)

            self.sampler_object_dict["Polled"] = self.polled_sampler

            if self.samplers_list.index("Polled") == 0:
                self.current_sampler = self.polled_sampler
                self.current_sampler_name = "Polled"

        if "HUF" in self.samplers_list:
            self.huf_sampler = HierarchicalUniform.HierarchicalUniform(agent_id_class_map,
                                                                       agent_class_action_domains,
                                                                       agent_class_observation_domains,
                                                                       auxiliary_data,
                                                                       config)

            self.sampler_object_dict["HUF"] = self.huf_sampler
            if self.samplers_list.index("HUF") == 0:
                self.current_sampler = self.huf_sampler
                self.current_sampler_name = "HUF"

        for sampler in self.samplers_list:
            if "TDF" in sampler or "BUF" in sampler:
                auxiliary_data.derived_hierarchy = hierarchy_map[sampler][0]
                auxiliary_data.derived_observation_domain = hierarchy_map[sampler][1]

                sampler_obj = FlattenedPolledSampler.FlattenedPolledSampler(agent_id_class_map,
                                                                           agent_class_action_domains,
                                                                           agent_class_observation_domains,
                                                                           auxiliary_data,
                                                                           config)

                sampler_obj.hierarchy = hierarchy_map[sampler][2]
                sampler_obj.hierarchical_action_domain = hierarchy_map[sampler][3]
                sampler_obj.h_obs_domain = hierarchy_map[sampler][4]

                sampler_obj._make_policy_groups(
                                sampler_obj.agent_id_class_map,
                                sampler_obj.agent_class_action_domains,
                                sampler_obj.agent_class_observation_domains)

                self.sampler_object_dict[sampler] = sampler_obj

                if self.samplers_list.index(sampler) == 0:
                    self.current_sampler = sampler_obj
                    self.current_sampler_name = sampler

        # Maps each sampler name to a list of indexes for which a sample was collected with that given sampler
        self.sample_index_map = {}
        self.sampler_index_list = []
        self.sampler_occurance_dict = {}
        if self.check_dist:
            self.sampler_dist_dict = {}

        for i in self.samplers_list:
            self.sample_index_map[i] = []
            self.sampler_occurance_dict[i] = 0

            if self.check_dist:
                self.sampler_dist_dict[i] = {}

        self.sample_index = 0

        self.sampler_occurance_dict[self.current_sampler_name] += 1
        self.last_sampler = None

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

    # This method uses the current node from the global stack for each agent
    def get_actions(self, observations: Dict[int, np.ndarray], use_max=False, append=True) -> Dict[int, np.ndarray]:
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

        for agent_id, obs in observations.items():
            if self.last_sampler:
                self.last_sampler.current_reward_dict[agent_id][self.action_stack_dict[agent_id][-1]] = self.current_reward_dict[agent_id][self.action_stack_dict[agent_id][-1]]
                self.last_sampler.add_s_prime(obs, agent_id)
                if append:
                    self.per_sample_derived.append(self.last_sampler.per_sample_derived[-1])
                    self.current_ep_derived.append(self.last_sampler.per_sample_derived[-1])
        all_actions = self.current_sampler.get_actions(observations, use_max, add_s_prime=False)

        for agent_id, action in all_actions.items():
            pg = self.agent_policy_groups_map[agent_id][0]
            abstracted_observation = self.slice_observation(pg, observations[agent_id], agent_id)
            self._update_oar(agent_id, pg.pg_id, abstracted_observation, action, 0)

        self.sample_index_map[self.current_sampler_name].append(self.sample_index)
        self.sampler_index_list.append(self.current_sampler_name)
        self.sample_index += 1

        self.abstracted_obs_dict = self.current_sampler.abstracted_obs_dict
        self.current_action_dict = self.current_sampler.current_action_dict
        self.current_reward_dict = self.current_sampler.current_reward_dict

        for k,v in self.abstracted_obs_dict[agent_id].items():
            cf = self.completion_function_pg[agent_id][k]
            self.abstracted_obs_dict[agent_id][k] = self.slice_observation(cf, obs, agent_id)

        self.action_stack_dict = self.current_sampler.action_stack_dict

        self.first_action = False
        self.last_sampler = self.current_sampler
        return all_actions

    def reset_sampler_index_list(self):
        self.sampler_index_list = []

    def reset_current_ep_derived(self):
        self.current_ep_derived = []

    def get_derived_samples(self, last_obs, sucessful_traj):
        """
        Returns the derived samples
        :return:
        """
        if not sucessful_traj:
            self.current_sampler.add_s_prime(last_obs[0], 0)
        else:
            self.current_sampler.add_s_prime(None, 0)

        self.per_sample_derived.append(self.current_sampler.per_sample_derived[-1])
        self.current_ep_derived.append(self.current_sampler.per_sample_derived[-1])

        inhibited_samples = []
        abstracted_samples = []

        for sampler in self.sampler_object_dict.values():
            inhibited_samples.extend(sampler.inhibited_samples[0])
            abstracted_samples.extend(sampler.abstracted_samples[0])

        return inhibited_samples, abstracted_samples

    def reset(self):
        '''
        Resets sampler for next episode
        :return:
        '''

        super(BOSS, self).reset()
        self.first_action = True
        self._init_derived()
        self.last_sampler = None

        if (self.display_distribution or self.kl_div) and sum(self.state_action_visits) > 0:
            self.distributions.append(self.state_action_visits/sum(self.state_action_visits))

    def check_all_agent_termination(self, observations: np.array):
        pass

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

    def update_current_sampler(self, current_sampler: str):
        """
        Sets the current sampler
        :return:
        """
        self.current_sampler_name = current_sampler
        try:
            self.current_sampler = self.sampler_object_dict[current_sampler]
        except KeyError:
            print("Requested sampler not in sampler object dict")

    def set_next_sampler(self):
        """
        Rotates samplers for the intial phase of the boss algorithm
        :return:
        """
        for ind, i in enumerate(self.samplers_list):
            if i == self.current_sampler_name:
                ind = (ind + 1)%len(self.samplers_list)
                break

        self.current_sampler_name = self.samplers_list[ind]
        self.current_sampler = self.sampler_object_dict[self.current_sampler_name]
        self.sampler_occurance_dict[self.current_sampler_name] += 1

    def get_per_sample_derived(self):
        return self.per_sample_derived
