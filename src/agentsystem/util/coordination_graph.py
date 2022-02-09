"""
A coordination graph object that maps edges to policy groups.

It performs validation of a coordination graph provided a list of agents,
    optionally ensuring that all agents belong to some part of the graph.

This class handles recalculating graph data such as adjacency lists when possible.
"""
from typing import List, Dict, Tuple, Any

import copy
import numpy as np

from agentsystem.util import PolicyGroup
from common.vecmath import CachedOps


class CoordinationGraph(object):

    @classmethod
    def validate_config(cls, graph):
        """
        A function which checks that a coordination graph list of edges is valid.
        :param graph:
        :return:
        """
        ret = isinstance(graph, list) and len(graph) > 0
        seen = set()
        for edge in graph:
            edge = tuple(edge)
            assert len(edge) == 2, 'Graphs have edges of pairs of agents.'
            assert isinstance(edge[0], int) and isinstance(edge[1], int), 'Graph edges are agents.'
            assert edge not in seen, 'Graph cannot have multi-edges'
            seen.add(edge)
        return ret

    def __init__(self,
                 agent_id_class_map: Dict[int, int],
                 coordination_graph: List[List[int]],
                 strict=False):
        """
        Initialize the CoordinationGraph helper object.
        :param agent_ids: Agent IDs the coordination graph will be defined on.
        :param coordination_graph: Coordination Graph expressed as a list of lists which describe edges.
        :param agent_id_class_map: A map of agent to agent classes. Useful for operations such as finding the
                                 joint class of a edge.
        :param strict: If true, makes sure that the graph has all existent and no nonexistent agents.
        """
        # Private graph data may include "ghost edges" which are empty.
        self._edges = copy.copy(coordination_graph)
        for i, edge in enumerate(self._edges):
            self._edges[i] = sorted(edge)
        agent_ids = [a1 for a1 in agent_id_class_map.keys()]

        self.agent_id_class_map = copy.deepcopy(agent_id_class_map)

        # Map edge ID to PG
        self.edge_pg_map: Dict[int, PolicyGroup] = {}
        # Map vertex ID to PG (vertex ID is (-1-agent_id))
        self.vertex_pg_map: Dict[int, PolicyGroup] = {}
        # Map agent to list of neighbor agents
        self.agent_neighbors: Dict[int, List[int]] = {agent_id: [] for agent_id in agent_ids}
        # Map agent to list of neighbor edge IDs
        self.agent_edges: Dict[int, List[int]] = {agent_id: [] for agent_id in agent_ids}
        # Map agent to list of neighbor edge IDs, the neighbor on that edge, and the primacy (agent is a1).
        self.agent_edges_info: Dict[int, List[Tuple[int, int, bool]]] = {agent_id: [] for agent_id in agent_ids}
        # Degree (of agent)
        self.agent_degree: Dict[int, int] = {agent_id: 0 for agent_id in self.agent_id_class_map}

        # Optionally validate that coordination graph does not contain invalid agents
        vertices = set()
        for edge_id, edge in enumerate(coordination_graph):
            a1, a2 = edge
            if strict:
                assert a1 in agent_ids, 'Invalid coordination graph with nonexistent agent [{}]'.format(a1)
                assert a2 in agent_ids, 'Invalid coordination graph with nonexistent agent [{}]'.format(a2)
            vertices.add(a1)
            vertices.add(a2)
            self.agent_edges[a1].append(edge_id)
            self.agent_edges[a2].append(edge_id)
            # Don't support directed graphs. Coordination (as this asys understands) is undirected.
            self.agent_neighbors[a1].append(a2)
            self.agent_neighbors[a2].append(a1)
            self.agent_edges_info[a1].append((edge_id, a2, True))
            self.agent_edges_info[a2].append((edge_id, a1, False))
            self.agent_degree[a1] += 1
            self.agent_degree[a2] += 1
        if strict:
            for agent_id in agent_ids:
                assert agent_id in vertices, 'Agent [{}] not in the coordination graph!'.format(agent_id)

        # Exposed edges
        self.edges = [edge for edge in self._edges]
        self.edge_ids = [i for i in range(len(self._edges))]
        self.labeled_edges = [(i, edge) for i, edge in enumerate(self._edges)]

    def __getitem__(self, edge_id: int) -> List[int]:
        """
        Allow the edges to be accessed directly by dict entry getting.
        :param edge_id:
        :return:
        """
        assert edge_id in self.edge_ids, 'Attempted to access removed edge [{}]'.format(edge_id)
        return self.edges[edge_id]

    def assign_edge_pg(self, edge_id: int, policy_group: PolicyGroup):
        """
        Assign the policygroup to the edge defined by the ID.
        """
        self.edge_pg_map[edge_id] = policy_group

    def assign_vertex_pg(self, agent_id: int, policy_group: PolicyGroup):
        """
        Assign the policygroup to the node defined by the agent ID.
        """
        self.vertex_pg_map[agent_id] = policy_group

    def get_edge_class(self, edge_id: int) -> Tuple[Any, Any]:
        """
        Get the sorted joint class of the agents on an edge.
        :return:
        """
        a1, a2 = self[edge_id]
        a1_class = self.agent_id_class_map[a1]
        a2_class = self.agent_id_class_map[a2]
        joint_class: Tuple[Any, Any] = tuple(sorted([a1_class, a2_class]))
        return joint_class

    def remove_agent(self, agent_id) -> List[PolicyGroup]:
        """
        Remove the agent with the specified agent ID.

        Returns removed policy groups for the caller to optionally handle.
        :param agent_id:
        :return:
        """
        # Destroy edges while preserving indices
        deleted_pgs = []
        for edge_id, neighbor, primacy in self.agent_edges_info[agent_id]:
            edge = self._edges[edge_id]
            self.edges.remove(edge)
            self.edge_ids.remove(edge_id)
            self.labeled_edges.remove((edge_id, edge))
            self._edges[edge_id] = []
            deleted_pgs.append(self.edge_pg_map[edge_id])
            del self.edge_pg_map[edge_id]
            self.agent_degree[neighbor] -= 1
        del self.agent_edges_info[agent_id]
        del self.agent_neighbors[agent_id]
        del self.agent_edges[agent_id]
        del self.agent_degree[agent_id]
        del self.agent_id_class_map[agent_id]

        # Vertex gets unceremoniously deleted
        # TODO (future, potential) could still store PG in case it could be reused in future
        if agent_id in self.vertex_pg_map:
            deleted_pgs.append(self.vertex_pg_map[agent_id])
            del self.vertex_pg_map[agent_id]

        return deleted_pgs

    def add_edge(self, edge_tup) -> int:
        """
        Add an edge (not an agent) to the graph.
        :param edge_tup: The tuple describing the new edge.
        :return:
        """
        missing_agent_msg = 'Agent [{}] in new edge does not exist in graph.'
        for a1 in edge_tup:
            assert a1 in self.agent_id_class_map, missing_agent_msg.format(a1)

        edge = list(edge_tup)

        # Edge ID is the same as policygroup id. The new policygroup will be assigned the new edge.
        new_e_id = len(self.edges)
        self._edges.append(edge)
        self.edges.append(edge)
        self.edge_ids.append(new_e_id)
        self.labeled_edges.append((new_e_id, edge))
        a1, a2 = edge_tup
        self.agent_neighbors[a1].append(a2)
        self.agent_neighbors[a2].append(a1)
        self.agent_edges[a1].append(new_e_id)
        self.agent_edges[a2].append(new_e_id)
        self.agent_edges_info[a1].append((new_e_id, a2, True))
        self.agent_edges_info[a2].append((new_e_id, a1, False))
        self.agent_degree[a1] += 1
        self.agent_degree[a2] += 1
        return new_e_id

    def __repr__(self):
        ret = 'Coordination graph: '
        ret += ', '.join([('edge:{}:pg{}'.format(edge, self.edge_pg_map[edge_id].pg_id)
                           if edge_id in self.edge_pg_map else '')
                        for edge_id, edge in self.labeled_edges])
        if self.vertex_pg_map:
            ret += '\n'
            ret += ', '.join(['node:{}:pg{}'.format(vertex_id, pg.pg_id) for vertex_id, pg in
                              self.vertex_pg_map.items()])
        return ret


class EliminationTree(object):
    """
    Special algorithmic construct for agent elimination algorithms where action-values are provided.

    Build tree over edges with (contents)[leaf/indexes-in-order] arrays, where leaf order remains consistent.
    (q,aX) just means that we have Q-values and separate output-action-map data for agent X, etc.
    We expand axes via leaves (combined leaves vs node leaves). Don't transpose.

    It's possible to greedily select the next agent to eliminate by the smallest size of the array, to greatly improve
        algorithmic efficiency.

    An example on a triangular graph:
    NODE 0 <== q[0,1], q[0,2]
            plus_leaves = [0,1,2]
            ==> q': q[0,1,2] = q[0,1,:] + q[0,:,2]
                a0: 0[1,2] = argmax_rnd_tie(q', axis=0)
            ==> q : q[1,2] = q'.max(axis=0)
    NODE 1 <== q[1,2], (q,a1)[1,2]
            plus_leaves = [1,2]
            ==> q': q[1,2] = q[1,2] + q[1,2]
                a1: 1[2] = argmax_rnd_tie(q', axis=0)
            ==> q : q[2] = q'.max(axis=0)
    NODE 2 <== (q,a1,a2)[2]
            plus_leaves = [2]
            ==> q': q[2] = q[2] (special when len(plus_leaves) == 1)
                a2: a2 = argmax_rnd_tie(q', axis=0)
            ==> SUMQ += q'.max(axis=0) (unnecessary in truth)
    Attribute 2 to action map
    Attribute 1 to action map using action map
    Attribute 0 to action map using action map
    """

    def __init__(self, graph: CoordinationGraph):
        """
        Initialize the elimination tree from the coordination graph.
        :param graph: Graph with which to initialize elimination tree.
        """
        tree = [(eid, edge) for eid, edge in graph.labeled_edges]
        agents = [aid for aid in graph.agent_id_class_map.keys() if len(graph.agent_neighbors[aid]) > 0]
        # Given an agent, have the order of the leaves subordinate to it
        self.agent_leaforder = {}
        # Given an agent, have the order of the data necessary to organize the subordinate leaf data
        self.agent_subleaves = {}
        # First agent to eliminate should be of least width
        self.agentorder = self.width_sort(agents, tree)
        # Construct the elimination tree
        for i in range(len(self.agentorder)):
            aid = self.agentorder[i]
            # Determine the subordinate leaves of the agent
            neighbors = [(label, edge) for label, edge in tree if aid in edge]
            leaves = []
            for node in neighbors:
                tree.remove(node)
                leaves.extend([subleaf for subleaf in node[1] if subleaf not in leaves])
            leaves.remove(aid)
            # Agent + leaves, for combining q-values
            plus_leaves = [aid] + leaves
            # Construct the leaf data manipulation objects
            neighbor_data = []  # (neighbor node id, slice_ext, transpose_order), or transform-func?
            for label, subleaves in neighbors:
                # Needs to know which node each neighbor is
                slice_ext = tuple([...] + [None] * (len(plus_leaves) - len(subleaves)))
                extended_subleaves = subleaves + [leaf for leaf in plus_leaves if leaf not in subleaves]
                transpose_order = [extended_subleaves.index(leaf) for leaf in plus_leaves]
                neighbor_data.append((label, slice_ext, transpose_order))
            # Append data to the tree and node
            tree.append((-aid-1, leaves))
            self.agent_leaforder[aid] = leaves
            self.agent_subleaves[aid] = neighbor_data
            if i < len(agents) - 1:
                self.agentorder = self.agentorder[:i + 1] + self.width_sort(self.agentorder[i + 1:], tree)

    def size(self):
        return sum([len(self.agent_subleaves[agent_id]) for agent_id in self.agentorder])

    def width_sort(self, agents, tree):
        # TODO actually sort according to width (e.g. with action size map info) instead of assuming homogeneity
        degrees = []
        for aid in agents:
            degrees.append(len([(label, edge) for label, edge in tree if aid in edge]))
        _, deg_sorted = zip(*sorted(zip(degrees, agents)))
        return deg_sorted

    def widths(self, action_size_map):
        """
        Return the size of arrays involved in calculation for each agent. (Diagnostic)
        :param action_size_map:
        :return:
        """
        ret = {}
        for aid in self.agentorder:
            neighbor_data = self.agent_subleaves[aid]
            leaforder = self.agent_leaforder[aid]
            q_shape = [action_size_map[aid]] + [action_size_map[agent] for agent in leaforder]
            q_size = np.prod(q_shape) * len(neighbor_data)
            ret[aid] = q_size
        return ret

    def argmax_shapes(self, action_size_map):
        """
        Return the shape of arrays involved in calculation for each agent. (Diagnostic)
        :param action_size_map:
        :return:
        """
        ret = {}
        for aid in self.agentorder:
            leaforder = self.agent_leaforder[aid]
            q_shape = [action_size_map[aid]] + [action_size_map[agent] for agent in leaforder]
            ret[aid] = q_shape
        return ret

    # @profile
    def agent_elimination(self, eqs):
        agent_action_funcs = {}
        # For each agent
        for aid in self.agentorder:
            neighbors = self.agent_subleaves[aid]
            # Calculate the expanded q-values (with max-q information from subordinate nodes)
            label, ext, order = neighbors[0]
            all_q = eqs[label][ext].transpose(order)
            for label, ext, order in neighbors[1:]:
                all_q = all_q + eqs[label][ext].transpose(order)
            # Take the argmax-random-tiebreak for the agent action function
            # And the rowmax for the max q values for this node to pass up the tree
            agent_action_funcs[aid], eqs[-aid - 1] = CachedOps.argmax_tiebreak(all_q, axis=0)
        # Determine action map
        action_map = {}
        for aid in reversed(self.agentorder):
            # Given the chosen action from upper agents, choose the actions for the lower agents
            indexing = tuple([action_map[leaf] for leaf in self.agent_leaforder[aid]])
            action_map[aid] = int(agent_action_funcs[aid][indexing])
        return action_map
