# __init__.py
"""
Agent systems module.
An agent system describes a system of agents.

In this framework, the agent system determines policy - not a singular agent.

The agent system takes cares of the agent updates - it determines how to update policy over the agents it handles, given the state given from the mdp.

The learner should update the policy for the agent system.
Depending on the agent system, multiple policies handled by the agent system might be updated.

Supported:
    independent
        Systems where the policy is agnostic to agent count / structure.

        IndependentAgentSystem
            Every single agent has its own independently learned policy.

        SharedSystem
            Every single agent uses the same policy to which all their updates broadcast.
            Should only be compatible with batchable algorithms and policies.....?
            Sounds like learning this one might be weird.

    dependent (Explicit Multiagent Learning)
        CoordinationGraphSystem
            Multiple agents with a joint policy factored over a coordination graph.

        LocalAgentSystem (future)
            Agents are distinguished by local, unique factorizations of State - AgentState.

        NSymmetricSystem (future)
            Factorization of global policy into inter-agent policy among partitioned n classes of agents.

This might be a single agent (SingleAgentSystem) or independent agents (IndependentAgentSystem)

"""
from .base import AgentSystem
from .IndependentSystem import IndependentSystem
from .SharedSystem import SharedSystem
from .CoordinationGraphSystem import CoordinationGraphSystem
from .NSymmetricSystem import NSymmetricSystem
