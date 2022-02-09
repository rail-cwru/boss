from copy import deepcopy

from typing import List

from domain.base import Domain
from domain.features import Feature


class ObservationDomain(Domain):
    """
    A Domain which describes possible Observations obtained from the environment,
    """
    # TODO get rid of num_agents somehow

    def __init__(self, items: List[Feature], num_agents: int=None):
        super(ObservationDomain, self).__init__(items)
        self.num_agents = num_agents
        self.items: List[Feature] = self.items

    @classmethod
    def join(cls, *domains):
        all_items = []
        domain_cls = type(domains[0])
        for i, domain in enumerate(domains):
            assert type(domain) == domain_cls, 'Cannot join Domains of different class, even inherited ones.'
            all_items += [item.copy_with_added_prefix(str(i)) for item in domain.items]
        # num_agents is superfluous and might be used in the future at a very low chance.
        # seriously, it's pointless here
        return domain_cls(all_items, None)
