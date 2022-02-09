"""
Properties module.

Contains base properties expected of Environment, AgentSystem, etc.

Properties are (unsurprisingly) properties that describe concrete classes.

These are used in the setup phase by the Config object during component initialization
    to ensure that no incompatible components are simultaneously loaded in an experiment.

What is a property?
    A property should determine compatibility.
    A quality which does not affect compatibility should not be a property.

    e.g.
        Whether or not a policy supports discrete observations is a property.
        Whether or not an environment contains trees is not a property, unless agentsystems somehow rely on that fact.

Properties that would affect compatibility but do not realistically do so also do not need to be implemented.
    e.g.
        Since all AgentSystems implicitly or explicitly support multi-agent learning, that support is not a property.
        However, since some AgentSystems do not support single-agent learning, that support is a property.
"""


class Properties(object):
    """
    Base properties class.
    """

    def __init__(self,
                 __sentinel=None,    # Used to make sure args are set via kwargs
                 use_single_agent=None,
                 use_coordination_graph=None,
                 use_function_approximator=None,
                 pytorch=None,
                 use_joint_observations=None,
                 use_discrete_action_space=None,
                 use_agent_deletion=None,
                 use_agent_addition=None,
                 use_hierarchy=None
                 ):
        """
        These are exclusionary properties.
        The default value is "None," meaning the component doesn't care about the property's value.

        A property value may be set, usually to a boolean, to indicate that a component requires a certain property:
        e.g. if an environment sets use_single_agent to true, indicating it only supports single agent simulation
             it will not work with agentsystem only supporting multiagent, as specified by use_single_agent=False.

        Some portions are strictly set by certain modules. This means that "don't care" shouldn't be available there.

        :param kwargs:
        """

        # Whether or not the component strictly requires or prohibits usage of a single agent
        self.use_single_agent = use_single_agent

        # Whether the component strictly requires or prohibits usage of coordination graph
        self.use_coordination_graph = use_coordination_graph

        # Whether the policy uses a function approximator.
        self.use_function_approximator = use_function_approximator

        # Uses pytorch
        self.pytorch = pytorch

        # Uses joint observations
        self.use_joint_observations = use_joint_observations

        # Only supports discrete action space
        self.use_discrete_action_space = use_discrete_action_space

        # Allow for addition of new agents
        self.use_agent_addition = use_agent_addition

        # Allow for deletion of agents
        self.use_agent_deletion = use_agent_deletion

        # Uses a hierarchy (e.g. in Hierarchical RL)
        self.use_hierarchy = use_hierarchy

    def check_required_are_satisfied(self):
        assert self.use_discrete_action_space is not None,\
            'use_discrete_action_space should always be set by some component in the experiment'

    def merge(self, properties: 'Properties') -> 'Properties':
        """
        Merge one properties object with another. Objects must be non conflicting
        """
        for attr, value in properties.__dict__.items():
            # check if both properties have a a same attribute not equal to none and not equal
            my_value = self.__getattribute__(attr)
            if value and my_value and (my_value != value):
                raise ValueError("Properties merge conflict: " + attr)
            elif value is not None and my_value is None:
                self.__setattr__(attr, value)
        return self

    def override_with(self, properties: 'Properties') -> 'Properties':
        """
        Override the values in this properties object with the not-none values in the passed-in properties object.
        """
        for attr, value in properties.__dict__.items():
            if value is not None:
                self.__setattr__(attr, value)
        return self
