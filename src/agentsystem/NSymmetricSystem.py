"""
NSymmetricSystem is an experimental extension of CoordinationGraphSystem where edges may have shared policies.

The NSymmetricSystem aims to reduce the size of CoordinationGraphSystem by exploiting symmetries inside the multiagent
    coordination graph. For example, consider an game of soccer where different agents with implicit roles might
    behave similarly with regard to each other.

    In addition, it aims to adapt to deletion by diffusing learned information about joint policies among many agents.

    It might be possible in the future to also exploit the symmetries of NSymmetricSystem to produce a new
        action selection algorithm.

A N-Symmetric System is defined by a degree of partition, n >= 1.

    This is specified in the configuration as "degree".

The policy-sharing method is structured as such:

    Walk through the coordination graph with depth-first traversal, assigning each agent an integer corresponding to
    the number of depths in the traversal where an agent of that agentclass was encountered.

    Take the modulo of the agent-assigned integers with n, the degree of partition.

    Label edges with the unordered duplet of the partitioned agent labels.

    Edges that share the same agent-class agents and the same duplet label will share policies.
"""
from typing import Dict, List, Tuple, Any

from agentsystem import CoordinationGraphSystem
from agentsystem.util import PolicyGroup
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from common.domain_transfer import DomainTransferMessage
from config import Config, checks
from config.config import ConfigItemDesc


class NSymmetricSystem(CoordinationGraphSystem):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return CoordinationGraphSystem.get_class_config() + [
            ConfigItemDesc(name='degree',
                           check=checks.positive_integer,
                           info='Degree of sharing (n) in the N-Symmetric System. '
                                '\n\t1: All edges with unique joint profiles share the same policy.'
                                '\n\t2 and above: Such edges share policy based on a graph coloring that assigns '
                                'each edge which would have shared a policy to n separate policies.')
        ]

    def __init__(self,
                 agent_id_class_map: Dict[int, int],
                 agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain],
                 auxiliary_data: Dict[str, Any],
                 experiment_config: Config):
        # Store values in agent partition
        self.partition_map: Dict[int, int] = {}

        # Map from agent-class and label duplet to Policy. Model is not shared, as with SharedSystem.
        self.shared_policy_map: Dict[Tuple[Any, Any, int, int]] = {}

        self.degree = experiment_config.find_config_for_instance(self).degree

        super().__init__(agent_id_class_map,
                         agent_class_action_domains,
                         agent_class_observation_domains,
                         auxiliary_data,
                         experiment_config)

    def _make_policy_groups(self, agent_class_map: Dict[int, int], agent_class_action_domains: List[ActionDomain],
                            agent_class_observation_domains: List[ObservationDomain]) -> List[PolicyGroup]:
        # Breadth-first traversal and labeling
        queue = []
        for a0 in self.agent_ids:
            ac_depth = {agent_class: 0 for agent_class in set(agent_class_map.values())}
            # Visit and push seed vertex.
            if a0 not in self.partition_map:
                ac0 = agent_class_map[a0]
                self.partition_map[a0] = ac_depth[ac0] % self.degree
                ac_depth[ac0] += 1
                queue.append(a0)
            # Then visit and push neighbors.
            while queue:
                a1 = queue.pop(0)
                visited_ac = set()
                for neighbor in self.graph.agent_neighbors[a1]:
                    if neighbor not in self.partition_map:
                        ac_neighbor = agent_class_map[neighbor]
                        self.partition_map[neighbor] = ac_depth[ac_neighbor] % self.degree
                        visited_ac.add(ac_neighbor)
                        queue.append(neighbor)
                for agent_class in set(agent_class_map.values()):
                    if agent_class in visited_ac:
                        ac_depth[agent_class] += 1
                    else:
                        ac_depth[agent_class] = 0

        return super(NSymmetricSystem, self)._make_policy_groups(agent_class_map,
                                                                 agent_class_action_domains,
                                                                 agent_class_observation_domains)

    def _make_joint_policy(self, edge_id):
        """
        Make the joint policy object for the pair of agents at the defined edge.
        :param edge_id: Edge we are making a policy for.
        :return:
        """
        # See if it already has been made.
        a1, a2 = self.graph[edge_id]
        ap1, ap2 = self.partition_map[a1], self.partition_map[a2]
        policy_identifier: Tuple[Any, Any, int, int] = self.graph.get_edge_class(edge_id) + tuple(sorted([ap1, ap2]))
        if policy_identifier in self.shared_policy_map:
            policy = self.shared_policy_map[policy_identifier]
        else:
            joint_act_dom, joint_obs_dom = self._make_joint_domains(edge_id)
            policy = self.policy_class(joint_obs_dom, joint_act_dom, self.top_config)
        self.shared_policy_map[policy_identifier] = policy
        return policy

    def transfer_domain(self, transfer_domain_message: DomainTransferMessage):
        # TODO - NSymmetricSystem should be different than CGS. We should be able to handle both agent addition and
        # removal by exploiting the shared policy map. However, this method is different from that of CGS, so
        # we will leave it for the (future) when this system is investigated.
        raise NotImplementedError()
