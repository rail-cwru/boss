from typing import List

from common import Trajectory
from policy import Policy
from model import Model


class PolicyGroup(object):
    """
    A PolicyGroup is a collection of agents over which a single policy and model should be learned and updated.

    These should be constructed by the AgentSystem.

    It is possible that many PolicyGroups may share a policy or model. However, this behavior will have to be defined
    in the AgentSystem.
    """

    def __init__(self,
                 pg_id: int,
                 agents: List[int],
                 policy: Policy,
                 model: Model,
                 max_time: int):
        """
        Instantiate the PolicyGroup object with the relevant parameters.
        :param pg_id: The unique ID.
        :param agents: The agent IDs belonging to this policy group
        :param policy: The policy belonging to this policy group
        :param model: The model associated with the PolicyGroup
        :param max_time: Max length of a single episode
        """
        self.pg_id = pg_id
        self.agents = agents
        self.policy = policy
        self.model = model

        # Allow growth?
        self.trajectory: Trajectory = None
        self.__max_time = max_time

        self.allocate_new_trajectory()

    def append(self, pg_obs, pg_act, pg_rew, done=False):
        pg_id = self.pg_id
        self.trajectory.append(pg_obs[pg_id], pg_act[pg_id], pg_rew[pg_id])
        self.trajectory.set_done(done)

    def allocate_new_trajectory(self):
        self.trajectory = Trajectory.allocate(self.__max_time, self.policy.domain_obs, self.policy.domain_act)
