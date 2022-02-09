from copy import deepcopy

import numpy as np

from domain.DiscreteActionDomain import DiscreteActionDomain
from domain.features import Feature
from domain.observation import ObservationDomain
from domain.actions import Action


class DiscreteActionFeatureDomain(ObservationDomain):
    """
    A domain describing features which are produced from a combination of Observation and Action data.
        That is, features for which "for each action, the feature changes."
        This is usable in some Q Learning and ADF Policies.

    The class contains a new attribute self.packed_shape = (ACT_SHAPE, FEAT_SHAPE).
        If the array is reshaped to this shape, the features per action will align in the second index.
        Thus, each row is a separate feature vectors, with the indexing of the rows matching the discrete action space.

    The class is initialized by DiscreteIndexedActionDomain and an ObservationDomain which should be populated with
        DomainItems describing the Action-tied joint data.
    """

    def __init__(self, features: ObservationDomain, act_domain: DiscreteActionDomain):
        self.raw_features = features
        self.actions = act_domain
        assert isinstance(act_domain, DiscreteActionDomain), "DiscreteActionFeatureDomain's action domain must" \
                                                             " be a DiscreteActionDomain."

        tiled_feature_items = []
        for i in range(act_domain.full_range):
            tiled_feature_items += [item.copy_with_added_prefix(str(i)) for item in features.items]
        super().__init__(items=tiled_feature_items)

        # [F, A] shape for use when reshaping is valuable.
        self.packed_shape = (np.prod(act_domain.ranges), features.shape[0])
        self.shape = (int(np.prod(self.packed_shape)), )
        self.discrete = True

    @classmethod
    def join(cls, *domains) -> 'DiscreteActionFeatureDomain':
        # Join DAFD. The obs features in each will be broadcast over the domains' widest action space.
        all_obs_domains = []
        # Which domain has the largest action space?
        widest_act_size = 0
        widest_act_domain = None
        # TODO - this is imprecise and not representative. We should actually make sure that
        #        features are broadcast properly along the components of joint actions.
        for domain in domains:  # type: DiscreteActionFeatureDomain
            if domain.actions.ranges[0] > widest_act_size:
                widest_act_domain = domain.actions
        exemplar = domains[0]
        for i, domain in enumerate(domains):
            assert type(domain) == type(exemplar),\
                'Cannot join Domains of different class ({} vs {}), even inherited ones.'\
                    .format(type(domain), type(exemplar))
            all_obs_domains += [domain.raw_features]
        # It will be up to the user to not join these inappropriately.
        # We don't make sure (for now) if the smaller ones have actions which exist in the greater ones.
        joint_obs_domain = ObservationDomain.join(*all_obs_domains)
        return DiscreteActionFeatureDomain(joint_obs_domain, widest_act_domain)

    @classmethod
    def broadcast_joint(cls, data1: np.ndarray, data2: np.ndarray):
        """
        Broadcast 2D action features to joint action features.
        MUST follow REPEAT1 TILE2 convention.
        Non-first axes must be same for both arrays.
        :param data1: Data from agent 1
        :param data2: Data from agent 2
        :return: Broadcasted data1, data2
        """
        return (
            np.repeat(data1, data2.shape[0], axis=0),
            np.tile(data2, [data1.shape[0], 1])
        )
