"""
A basic information container class for passing auxiliary info (e.g. joint observation domains, hierarchies)
    from environment to agentsystem during initialization.

Since using dict is unsafe and does not support IDE tools (like checking for invalid fields),
    this class can contain the appropriate fields and be updated accordingly, like the `Properties`.
"""


class AuxiliaryEnvInfo(object):

    def __init__(self,
                 joint_observation_domains=None,
                 hierarchy=None,
                 derived_hierarchy=None,
                 derived_observation_domain=None
                 ):
        self.joint_observation_domains = joint_observation_domains
        self.hierarchy = hierarchy
        self.derived_hierarchy = derived_hierarchy
        self.derived_observation_domain = derived_observation_domain
