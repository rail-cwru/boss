
import numpy as np
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from agentsystem import IndependentSystem
import pickle as pkl
from typing import Dict, List, Tuple, Any
from policy.function_approximator.basis_function.ExactBasis import ExactBasis
from domain.conversion import FeatureConversions
from config import Config, AbstractModuleFrame, ConfigItemDesc, checks


class RandomSampler(IndependentSystem, AbstractModuleFrame):
    """
    An Agent System designed for flat, random sampling for offline learning
    Not intended to be used as an online agent system, the results of this may not be as expected
    @author Eric Miller
    @contact edm54@case.edu
    """

    def get_class_config() -> List[ConfigItemDesc]:
        return[
                ConfigItemDesc('optimistic', checks.boolean, 'Use balanced wandering', default=False, optional=True)
        ]

    def __init__(self,
                 agent_id_class_map: Dict[int, int],
                 agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain],
                 auxiliary_data: Dict[str, Any],
                 config: Config):

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

        self.num_states = [domain_item.num_values() for domain_item in
                           agent_class_observation_domains[agent_id_class_map[0]].items]

        self.num_actions = agent_class_action_domains[0].full_range
        self.state_action_basis = ExactBasis(np.asarray(self.num_states), self.num_actions)
        self.state_action_visits = np.zeros(self.state_action_basis.size(), np.int64)

        if hasattr(config, 'display_distribution'):
            self.display_distribution = config.display_distribution
            self.distributions = []
        else:
            self.display_distribution = False

        self.optimistic = config.sampler.optimistic if hasattr(config.sampler, 'optimistic') else False
        print("Use balanced wandering:", self.optimistic)

        super(IndependentSystem, self).__init__(agent_id_class_map, agent_class_action_domains,
                                                agent_class_observation_domains, auxiliary_data, config)

    def load_target_dist(self):
        file_name = 'heist_target.list'

        with open(file_name, 'rb') as target_dist_file:
            self.target_distribution = pkl.load(target_dist_file)
        print('Loaded', file_name)

    def get_actions(self, observations: Dict[int, np.ndarray], use_max: bool = False) -> Dict[int, np.ndarray]:
        """
        :return: a random action from the action domain
        """
        # Actions according to agent class.
        all_actions = {}
        for agent_id, observation in observations.items():
            pg = self.agent_policy_groups_map[agent_id][0]

            if self.optimistic:
                all_actions[agent_id] = self.sample_optimistic_action(observation, agent_id)
            else:
                all_actions[agent_id] = self.random_action(agent_id)

            if self.kl_div:
                mapped_obs = self._map_table_indices(observation, pg.policy)
                index = self.state_action_basis.get_state_action_index(mapped_obs, all_actions[agent_id][0])
                self.state_action_visits[index] += 1

        return all_actions

    def sample_optimistic_action(self, obs, agent_id):
        pg = self.agent_policy_groups_map[agent_id][0]
        visits = np.zeros(self.num_actions)
        indicies = []

        # Get indexes of states/actions from basis
        for action in range(self.num_actions):

            if len(obs) != len(self.num_states):
                mapped_obs = self._map_table_indices(obs, pg.policy)
            else:
                mapped_obs = obs[:]

            sa_index = self.state_action_basis.get_state_action_index(mapped_obs, action)
            indicies = np.append(indicies, sa_index)
            num_visits = self.state_action_visits[sa_index]

            # Get number of occurances of each action from the table
            visits[action] = num_visits

        # Randomly select among minimally visited states
        action = [np.random.choice(np.where(visits == visits.min())[0])]

        index = self.state_action_basis.get_state_action_index(mapped_obs, action[0])
        self.state_action_visits[index] += 1

        return action

    def random_action(self, agent_id):
        pg = self.agent_policy_groups_map[agent_id][0]
        action = [pg.policy.domain_act.random_sample()]
        return action

    def get_distribution(self):
        return self.state_action_visits/sum(self.state_action_visits)

    def _map_table_indices(self, states: np.ndarray, policy) -> Tuple[int, ...]:
        """
        Map a single state from native representation to table index.
        :param states: State in native representation
        :return: Table index
        """
        mapped_state = []
        for feature in policy.domain_obs.items:
            domain_state = policy.domain_obs.get_item_view_by_item(states, feature)
            domain_mapped_state = FeatureConversions.as_index(feature, domain_state)
            mapped_state.append(domain_mapped_state)
        return np.asarray(mapped_state)

    def reset(self):
        '''
        Resets sampler for next episode
        :return:

        '''
        super(RandomSampler, self).reset()
        if self.display_distribution and sum(self.state_action_visits) > 0:
            self.distributions.append(self.state_action_visits/sum(self.state_action_visits))
