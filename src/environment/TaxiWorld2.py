
from typing import Dict, List, Tuple, TYPE_CHECKING, Union
from domain.features import CoordinateFeature, VectorFeature, DiscreteFeature, Feature
from domain.observation import ObservationDomain
from environment.TaxiWorld import TaxiWorld
import numpy as np

if TYPE_CHECKING:
    from config import Config


class TaxiWorld2(TaxiWorld):
    """
    This is an alternate version of the original Taxiworld that has a more efficient state representation
    This updated state representation matches that of the MaxQ paper
    3600 state-action pairs
    Designed with hierarchy in mind, but can be used flat too
    """

    def __init__(self, config: 'Config'):

        config = config.find_config_for_instance(self)
        super(TaxiWorld2, self).__init__(config)

    def observe(self, obs_groups=None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        """
        :return:
        """
        observations = {}

        # Other than the taxi location, every agent gets the same observation
        pre_observation = self._observation_domain.generate_empty_array()

        for p, source in self._passenger_sources.items():
            p_source_slice = self._observation_domain.index_for_name(name=f'{p}', prefix='passenger_loc')
            # Passenger in taxi
            if self._passenger_picked_up[p] > -1:
                pre_observation[p_source_slice] = 0

            # Passenger dropped off
            elif self._passenger_picked_up[p] < -1:
                pre_observation[p_source_slice] = 1
            # Passenger at source
            else:
                pre_observation[p_source_slice] = self._sd_locations.index(source) + 2

        for p, dest in self._passenger_destinations.items():
            p_dest_slice = self._observation_domain.index_for_name(name=f'{p}', prefix='destination')
            pre_observation[p_dest_slice] = [dest[0], dest[1]]

        observation = pre_observation.copy()
        for agent in self.agent_id_list:
            taxi_loc_slice = self._observation_domain.index_for_name(name='taxi_loc')
            pos = self._agent_positions[agent]
            observation[taxi_loc_slice] = [pos[0], pos[1]]
            observations[agent] = observation

        return observations

    def _create_observation_domains(self, config) -> Dict[int, ObservationDomain]:
        """
        Observation domain contains the state variables for Taxi World
        Every taxi can see every other taxi, passenger, and passenger status
        holding_passenger takes several values:
            -1 is initial value, passenger has not been picked up yet
            0...n-1 is the taxi that is currently holding the passenger
            -2 ... -n - 1 is the taxi that was holding the passenger before it was dropped off
                NOTE: -2 is taxi 0
        :param config:
        :return:
        """

        items = []
        # The agent's position
        taxi_loc = CoordinateFeature(name='taxi_loc', lower=[0, 0],
                                     upper=[self.rows, self.cols],
                                     #upper=[self.cols, self.rows],
                                     is_discrete=True)
        items.extend([taxi_loc])

        # The list of passenger sources
        for p in range(self.num_passengers):
            # TODO fix type hinting here: https://stackoverflow.com/questions/53974936/pycharm-and-type-hinting-warning
            # 0 = passenger in taxi, 1 = passenger dropped off
            p_loc = DiscreteFeature(name=f'{p}', size = len(self._sd_locations) + 2,
                                    starts_from=0, prefix='passenger_loc')
            items.extend([p_loc])

        # The list of passenger destinations
        for p in range(self.num_passengers):
            p_loc = CoordinateFeature(name=f'{p}', lower=[0, 0],
                                      upper=[self.rows, self.cols],
                                      is_discrete=True, sparse_values=self._sd_locations,
                                      prefix='destination')

            items.extend([p_loc])

        self._observation_domain = ObservationDomain(items, num_agents=self.n)
        return {self.agent_class: self._observation_domain}

    def abstracted_observation_domain(self, state_variables: set) -> ObservationDomain:
        """
        Observation domain contains the state variables for Taxi World
        Every taxi can see every other taxi, passenger, and passenger status
        Holds only the state variables that are required at this node in the hierarchy
        :return: Observation Domain
        """
        items = []

        if 'taxi_loc' in state_variables:
            # The agent's position
            taxi_loc = CoordinateFeature(name='taxi_loc', lower=[0,0],
                                         upper=[self.rows, self.cols],
                                         #upper=[self.cols, self.rows],
                                         is_discrete=True)
            items.extend([taxi_loc])

        if 'passenger_loc' in state_variables:
            # The list of passenger sources
            for p in range(self.num_passengers):
                p_loc = DiscreteFeature(name=f'{p}', size=len(self._sd_locations) + 2,
                                        starts_from=0, prefix='passenger_loc')
                items.extend([p_loc])

        if 'destination' in state_variables:
            # The list of passenger destinations
            for p in range(self.num_passengers):
                p_loc = CoordinateFeature(name=f'{p}', lower=[0, 0],
                                          upper=[self.rows, self.cols],
                                          #upper=[self.cols, self.rows],
                                          is_discrete=True, sparse_values=self._sd_locations,
                                          prefix='destination')

                items.extend([p_loc])

        return ObservationDomain(items, num_agents=self.n)
