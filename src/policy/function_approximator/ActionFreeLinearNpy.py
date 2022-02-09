from typing import Dict, List
import numpy as np

from common import TrainingSamples, Properties

# TODO harmful to import named module like this
from domain.DiscreteActionFeatureDomain import DiscreteActionFeatureDomain
from model.BasicEligibilityModel import BasicEligibilityModel
from policy.function_approximator import FunctionApproximator
from domain import ObservationDomain, ActionDomain
from config import Config, ConfigItemDesc


class ActionFreeLinearNpy(FunctionApproximator):
    """
    See ActionFreeLinear - This is a Numpy variant which is faster for online algorithms. (?!)

    # TODO heavy refactor
    """

    @classmethod
    def properties(cls) -> 'Properties':
        return Properties(pytorch=False)

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return []

    def __init__(self, config: Config, domain_obs: ObservationDomain, output_size: int):
        assert isinstance(domain_obs, DiscreteActionFeatureDomain), 'ActionFreeLinear (Numpy) only works with ' \
                                                                    'DiscreteActionFeatureDomain.'
        self.action_shape, self.feature_size = domain_obs.packed_shape
        FunctionApproximator.__init__(self, config, domain_obs, output_size)
        # AttributableFA.__init__(self)
        self.weights = np.full([self.feature_size], 0.1, dtype='float32')

    def compile(self, loss_func, learning_rate):
        self.loss = loss_func
        self.learning_rate = learning_rate

    def get_eligibility_trace(self, eligibility_model: BasicEligibilityModel, observations: np.ndarray, mode=None):
        """
        Trace model for AFL function approximator
        :param eligibility_model: BasicEligibilityModel keeping track of eligibility traces which trace will be added to
        :param observations: Observation for episode and policy group
        :return: eligibility trace for new time step
        """
        # TODO consider how to refactor in future?
        if eligibility_model.n_eligs < 1:
            return observations[-1]
        else:
            learning_rate = self.config.algorithm.learning_rate
            prev_elig = eligibility_model.eligibilities[-1].reshape(self.action_shape, -1)
            prev_feat = observations[-2].reshape(self.action_shape, -1)
            discount_term = self.config.algorithm.lmbda * self.config.algorithm.discount_factor
            # Default term
            new_eligibility = discount_term * prev_elig
            # Term defined by derivative?
            new_eligibility += prev_feat
            # Term defined by derivative
            new_eligibility -= learning_rate * discount_term * np.dot(np.dot(prev_elig, prev_feat.T), prev_feat)
            return new_eligibility.ravel()

    def get_trace_size(self, mode=None):
        """
        :param mode: not used currently
        :return: Size of input features
        """
        return self.input_size,

    def get_variable_vals(self):
        return {'weights': self.weights}

    def set_variable_vals(self, vals: Dict):
        self.weights = vals['weights']

    # @profile
    def update_with_eligibility(self, policy, errors, value_diff, curr_eligibilities: np.ndarray, actions,
                                observations):
        # TODO: Will need to be adapted for multiple actions / non-indexable actions
        # Shape: [N_STEP, ACTIONFEATURE]
        update = (errors + value_diff)[:, None] * curr_eligibilities - value_diff[:, None] * observations[:-1]
        # Produce the batched updates for the actions existing in trajectory
        actions_taken = actions[:-1, 0]
        # Shape: [N_STEP, ACTION, FEATURE]
        update = update.reshape(-1, self.action_shape, self.feature_size)
        # Shape: [N_STEP, FEATURE]
        update = update[np.r_[:len(actions_taken)], actions_taken]
        # This is the minus gradient which is ADDED to produce update
        plus_grad = self.config.algorithm.learning_rate * np.sum(update, axis=0)
        self.weights += plus_grad

    def update(self, dataset: TrainingSamples, *args):
        raise NotImplementedError('Offline updates not implemented for AFL (Numpy) FA.')

    def eval(self, feature_vectors) -> np.ndarray:
        if len(feature_vectors.shape) == 1:
            feature_vectors = np.expand_dims(feature_vectors, 0)
        feature_vectors = feature_vectors.reshape((-1, self.action_shape, self.feature_size))
        return (self.weights[None, None, :] * feature_vectors).sum(axis=2)

        # return sess.run([self.out], feed_dict={self.input: feature_vectors})[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
        return

