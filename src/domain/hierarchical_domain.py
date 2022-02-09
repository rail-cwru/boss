from typing import Optional, Dict, Set, List, Tuple, Iterable, Union, Any

from domain.DiscreteActionDomain import DiscreteActionDomain
from domain.actions import Action, DiscreteAction

__all__ = ['HierarchicalActionDomain', 'ParameterizedActionHierarchy', 'ExpandedActionHierarchy',
           'action_hierarchy_from_config']


class HierarchicalActionDomain(DiscreteActionDomain):
    """
    @author Robbie Dozier
    @contact grd27@case.edu

    An ActionDomain built to store hierarchical information. Subclass of DiscreteActionDomain, so instances of this
    class can still be used in flat scenarios because calling methods on this object directly will ignore the hierarchy.
    For example, self.random_sample() will sample a random primitive action.

    To use this class for a hierarchical subtask, use the domain_for_action() method. For example, in TaxiWorld using
    domain_for_action('Root') will return a DiscreteActionDomain instance with two possible actions (Get and Put).
    __getattr__() is overloaded to call domain_for_action().
    """
    def __init__(self, name: str, expanded_action_hierarchy: 'ExpandedActionHierarchy'):
        """
        @param name: Name to give the DiscreteAction which contains all the primitive actions
        @param expanded_action_hierarchy: ExpandedActionHierarchy object representing the current hierarchy.
        """
        self._action_hierarchy = expanded_action_hierarchy

        # Construct list of actions to pass
        leaf_actions = expanded_action_hierarchy.get_leaf_nodes()
        action = DiscreteAction(name, len(leaf_actions))
        super(HierarchicalActionDomain, self).__init__([action], 1)
        self.root_name = expanded_action_hierarchy.root
        # Construct subtask action domains
        self.__action_domain_dict = {}
        for action_name in expanded_action_hierarchy.get_non_leaf_nodes():
            actions = expanded_action_hierarchy.get_children(action_name)
            action = DiscreteAction(action_name, len(actions))
            self.__action_domain_dict[action_name] = DiscreteActionDomain([action], 1)

        # store action domain of primitives for value function
        for action_name in expanded_action_hierarchy.get_leaf_nodes():
            action = DiscreteAction(action_name, 1)
            self.__action_domain_dict[action_name] = DiscreteActionDomain([action], 1)

    def domain_for_action(self, action: str) -> DiscreteActionDomain:
        """
        @param action: Name of new root subtask.
        @return: A new HierarchicalActionDomain instance at "action"
        """
        return self.__action_domain_dict[action]

    def __getattr__(self, action: str) -> DiscreteActionDomain:
        return self.domain_for_action(action)

    @property
    def action_hierarchy(self) -> 'ExpandedActionHierarchy':
        """
        @return: The action hierarchy. Note that only the subgraph of the overall hierarchy with root_name as root is
        stored.
        """
        return self._action_hierarchy


def action_hierarchy_from_config(config: Dict[str, dict]) -> 'ParameterizedActionHierarchy':
    """
    Takes in the dict from an action hierarchy config and generates an ParameterizedActionHierarchy instance and
    constructs the associated action hierarchy graph.

    @param config: Config dict from an action hierarchy.
    @return: An ParameterizedActionHierarchy instance.
    """
    state_variables = set(config['state_variables'])
    primitive_action_map = config['primitive_action_map']
    actions = config['actions']
    action_hierarchy = ParameterizedActionHierarchy(actions, state_variables, primitive_action_map)
    # Build edges
    for action_name in actions.keys():
        if 'children' in actions[action_name]:
            for child, bound_variables in actions[action_name]['children'].items():
                action_hierarchy.bind_subtask(action_name, child, bound_variables)
    return action_hierarchy


class ParameterizedActionHierarchy(object):
    """
    @author Robbie Dozier
    @contact grd27@case.edu

    A graph data structure which represents an action hierarchy. In this graph, the nodes are the hierarchy's actions,
    and the edges are state variables bound to that task-subtask pair. For example, the TaxiWorld domain is like this:

                                                 Root
                                            /             \
                                        Get                Put
                                  /  ['source'] \  / ['destination']  \
                            Pickup            Navigate                 Putdown
                                           /   /   \   \
                                     North  South East  West

    Note how 'source' is bound to Navigate from Get, and 'destination' is bound to Navigate from Put, which means that,
    for example, when Get calls Navigate, it will call Navigate with the current value of 'source'-- so if 'source' is
    "x", it will call Navigate(x).

    All actions, state variables, etc. must be included at instantiation time. Edges are added through bind_subtask().
    """
    def __init__(self, actions: Dict[str, dict], state_variables: Iterable[str], primitive_action_map: Dict[str, int]):
        """
        @param actions: Dict of form {'action name': {dict from config}}.
        @param state_variables: Iterable containing the names of each state variable. Will be cast to a set.
        @param primitive_action_map: Dict of form {'action name': int value of action}
        """
        self._actions = actions
        self._edges = {}
        self.state_variables = set(state_variables)
        self.primitive_action_map = primitive_action_map

    def bind_subtask(self, parent: str, child: str, bound_variables: Optional[List[str]] = None):
        """
        Creates a new edge in the graph.

        @param parent: Name of parent action.
        @param child: Name of child action.
        @param bound_variables: Optional list of variables bound to the edge, see class docstring.
        """
        if bound_variables is None:
            bound_variables = []
        edge = bound_variables
        self._edges[(parent, child)] = edge

    def get_children(self, parent: str) -> List[Tuple[str, List[str]]]:
        """
        Gets the children of an action.

        @param parent: Name of parent action.
        @return: A list of the form [('child name', ['bound variables'])]
        """
        return [(c, e) for (p, c), e in self._edges.items() if parent == p]

    def get_parents(self, child: str) -> List[Tuple[str, List[str]]]:
        """
        Gets the parents of an action.

        @param child: Name of child action.
        @return: A list of the form [('parent name', ['bound variables'])]
        """
        return [(p, e) for (p, c), e in self._edges.items() if child == c]

    def is_flat_graph(self, root: Optional[str] = None) -> bool:
        """
        @param root: Node at which to test if the graph is flat. Defaults to the graph's root if none is specified.
        @return: Whether or not the graph is flat at a given node, i.e. all child nodes are leaf nodes.
        """
        if root is None:
            root = self.root

        for child, _ in self.get_children(root):
            if self.get_children(child):
                return False
        return True

    def get_leaf_nodes(self, root: Optional[str] = None) -> Set[str]:
        """
        @param root: Node from which to propagate. Defaults to the graph's root if none is specified.
        @return: Set containing all leaf nodes reachable from root.
        """
        if root is None:
            root = self.root

        if self.is_flat_graph(root):
            if self.get_children(root):
                return {name for name, _ in self.get_children(root)}
            return {root}
        leaves = set()
        for child, _ in self.get_children(root):
            leaves = leaves.union(self.get_leaf_nodes(child))
        return leaves

    def get_non_leaf_nodes(self, root: Optional[str] = None) -> Set[str]:
        """
        @param root: Node from which to propagate. Defaults to the graph's root if none is specified.
        @return: Set containing all non-leaf nodes reachable from root.
        """
        if root is None:
            root = self.root

        if self.is_flat_graph(root):
            if self.get_children(root):
                return {root}
            return set()
        leaves = set()
        for child, _ in self.get_children(root):
            leaves = leaves.union(self.get_non_leaf_nodes(child))
        return leaves

    def get_subgraph(self, new_root: str):
        """
        @param new_root: Node from which to propagate.
        @return: A new ParameterizedActionHierarchy instance containing all nodes and edges reachable from new_root,
        i.e. a new action hierarchy with new_root as the root action.
        """
        # First pass, get set of actions
        actions = {new_root: self.actions[new_root]}
        subgraph = ParameterizedActionHierarchy(actions, self.state_variables, self.primitive_action_map)
        for child, bound_variables in self.get_children(new_root):
            subgraph = self.union(subgraph, self.get_subgraph(child))
            subgraph.bind_subtask(new_root, child, bound_variables)
        return subgraph

    def assert_no_cycles(self):
        """
        @todo
        @raise RuntimeError if the graph is cyclic
        """
        # Find transitive closure
        raise NotImplementedError("I'll write this eventually")

    def compile(self, bound_variable_values: Dict[str, Set[str]],
                bound_var_map: Optional[Dict[str, Any]] = None):
        """
        Converts this ParameterizedActionHierarchy instance into an ExpandedActionHierarchy instance.

        @param bound_variable_values: Dict of the form {'state variable': {set of possible values}}. Used for expanding
        the graph and naming the parameterized actions.

        @param bound_var_map: Dict of form {'Action Prefix: 'Action suffix' : 'Bound var values'}
        For navigate, {'Navigate': {'_0_0':(0,0), '_4_4' : (4,4)...}

        @return: An ExpandedActionHierarchy representation of this action hierarchy.
        """
        return ExpandedActionHierarchy(self, bound_variable_values,
                                       self.state_variables,
                                       self.primitive_action_map,
                                       bound_var_map=bound_var_map)

    @property
    def root(self) -> str:
        """
        @return: The root of this graph, i.e. the first action with no parents.
        """
        for action in self.actions.keys():
            if not self.get_parents(action):
                return action

    @property
    def actions(self) -> Dict[str, dict]:
        """
        @return: Dict of form {'action name': {dict from config}}. DO NOT USE THIS TO FIND THE CHILDREN/PARENTS OF AN
        ACTION OR ANY OTHER INFORMATION THAT IS CONTAINED ELSEWHERE.
        """
        return self._actions

    @property
    def edges(self) -> Dict[Tuple[str, str], List[str]]:
        """
        @return: Dict of the form {('parent', 'child'), ['bound variables']}
        """
        return self._edges

    @staticmethod
    def union(graph_a: 'ParameterizedActionHierarchy', graph_b: 'ParameterizedActionHierarchy'):
        """
        Combines two ParameterizedActionHierarchy instances. Not idiotproofed so if you are an idiot use caution.

        @param graph_a: First ParameterizedActionHierarchy to be merged.
        @param graph_b: Second ParameterizedActionHierarchy to be merged.
        @return: ParameterizedActionHierarchy instance with the union of the nodes and edges.
        @raise: ValueError if the graphs have edges which disagree
        """
        # Make new action hierarchy with union of state variables and primitive action maps
        new_graph = ParameterizedActionHierarchy(
            dict(graph_a.actions, **graph_b.actions),
            graph_a.state_variables.union(graph_b.state_variables),
            dict(graph_a.primitive_action_map, **graph_b.primitive_action_map)
        )

        # This loop can be optimized probably but that's a job for another time
        for parent_a, child_a, bound_vars_a in graph_a.edges:
            for parent_b, child_b, bound_vars_b in graph_b.edges:
                if parent_a == parent_b and child_a == child_b and bound_vars_a != bound_vars_b:
                    raise ValueError("Same edge found with different bound variables")

        for parent, child, edge in graph_a.edges:
            new_graph.bind_subtask(parent, child, edge)
        for parent, child, edge in graph_b.edges:
            new_graph.bind_subtask(parent, child, edge)

        return new_graph


class ExpandedActionHierarchy(object):
    """
    @author Robbie Dozier
    @contact grd27@case.edu

    The ParameterizedActionHierarchy class is written such that a parameterized action, such as Navigate(x) in
    TaxiWorld, can be called with an arbitrary argument. In reality, each possible value of x corresponds to a different
    subtask which contains a different policy. Therefore, the ParameterizedActionHierarchy graph must be "compiled" into
    a graph which, instead of storing the bound variables on the edges, explicitly enumerates all possible arguments for
    the parameterized actions and constructs the corresponding graph.
    """
    def __init__(self, source_graph_or_actions: Union[ParameterizedActionHierarchy, Dict[str, dict]],
                 bound_variable_values_or_edges: Dict[str, Iterable[str]], state_variables: Iterable[str],
                 primitive_action_map: Dict[str, int], ground_var_map: Optional[Dict[str, Any]] = None,
                 bound_var_map: Optional[Dict[str, Any]] = None):
        """
        Creates a new ExpandedActionHierarchy object, either by receiving the actions and edges dictionaries or by being
        passed an ParameterizedActionHierarchy instance and the possible values of each state variable bound to an edge.

        @param source_graph_or_actions: Either an ParameterizedActionHierarchy instance or a dict of the form
        {'action name': {dict from config}}.
        @param bound_variable_values_or_edges: Either a dict of the form {'state variable': {set of possible values}} or
        a dict of the form {'parent action name': {iterable of child actions}}. This will be cast to a list.
        @param state_variables: Iterable containing the names of each state variable. Will be cast to a list.
        @param primitive_action_map: Dict of form {'action name': int value of action}
        @param bound_var_map: Dict of form {'Action Prefix: 'Action suffix' : 'Bound var values'}
        For navigate, {'Navigate': {'_0_0':(0,0), '_4_4' : (4,4)...}
        """
        if isinstance(source_graph_or_actions, ParameterizedActionHierarchy):
            source_graph = source_graph_or_actions
            bound_variable_values = bound_variable_values_or_edges

            root = source_graph.root
            self._actions, self._edges, self._ground_var_map = _build_graph(source_graph,
                                                                            {k: set(v) for k, v in
                                                                            bound_variable_values.items()},
                                                                            root, bound_var_map=bound_var_map)
        else:
            actions = source_graph_or_actions
            edges = bound_variable_values_or_edges
            self._actions, self._edges = actions, edges
            if ground_var_map is None:
                raise ValueError('Must define parameter key when manually creating ExpandedActionHierarchy')
            self._ground_var_map = ground_var_map
        self.primitive_action_map = primitive_action_map
        # Bind to a list to preserve ordering
        self._edges = {k: list(v) for k, v in self._edges.items()}
        self.state_variables = list(state_variables)

    def get_termination_predicate(self, node: str):
        return self._actions[node]['termination']

    def get_pseudo_rewards(self, node: str):
        if 'pseudo_r' in self._actions[node]:
            return self._actions[node]['pseudo_r']
        else:
            print('Primitive Actions have no Pseudo Rewards')
            return None

    @property
    def root(self) -> str:
        """
        @return: The root of this graph, i.e. the first action with no parents.
        """
        for action in self.actions.keys():
            if not self.get_parents(action):
                return action

    @property
    def actions(self) -> Dict[str, dict]:
        """
        @return: Dict of form {'action name': {dict from config}}. DO NOT USE THIS TO FIND THE CHILDREN/PARENTS OF AN
        ACTION OR ANY OTHER INFORMATION THAT IS CONTAINED ELSEWHERE.
        """
        return self._actions

    @property
    def ground_var_map(self) -> Dict[str, Any]:
        return self._ground_var_map

    @property
    def edges(self) -> Dict[str, List[str]]:
        """
        @return: Dict of the form {'parent': [list of children]}. Keep in mind that we no longer need to map data to
        each edge.
        """
        return self._edges

    def bind_subtask(self, parent: str, child: str):
        """
        Creates a new edge in the graph.

        @param parent: Name of parent action.
        @param child: Name of child action.
        """
        try:
            if child not in self._edges[parent]:
                self._edges[parent].append(child)
        except KeyError:
            self._edges[parent] = [child]

    def get_children(self, parent: str) -> List[str]:
        """
        Gets the children of an action.

        @param parent: Name of parent action.
        @return: A list containing the names of the child actions.
        """
        try:
            return self.edges[parent]
        except KeyError:
            return []

    def get_parents(self, child: str) -> Set[str]:
        """
        Gets the parents of an action.

        @param child: Name of child action.
        @return: A set containing the names of the parent actions.
        """
        return set([parent for parent in self.edges.keys() if self.edges[parent] == child])

    def is_flat_graph(self, root: str = None) -> bool:
        """
        @param root: Node at which to test if the graph is flat. Defaults to the graph's root if none is specified.
        @return: Whether or not the graph is flat at a given node, i.e. all child nodes are leaf nodes.
        """
        if root is None:
            root = self.root

        for child in self.get_children(root):
            if self.get_children(child):
                return False
        return True

    def get_leaf_nodes(self, root: Optional[str] = None) -> Set[str]:
        """
        @param root: Node from which to propagate. Defaults to the graph's root if none is specified.
        @return: Set containing all leaf nodes reachable from root.
        """
        if root is None:
            root = self.root

        if self.is_flat_graph(root):
            if self.get_children(root):
                return {name for name in self.get_children(root)}
            return {root}

        leaves = set()
        for child in self.get_children(root):
            leaves = leaves.union(self.get_leaf_nodes(child))

        return leaves

    def get_non_leaf_nodes(self, root: Optional[str] = None) -> Set[str]:
        """
        @param root: Node from which to propagate. Defaults to the graph's root if none is specified.
        @return: Set containing all non-leaf nodes reachable from root.
        """
        if root is None:
            root = self.root

        if self.is_flat_graph(root):
            if self.get_children(root):
                return {root}
            return set()

        nodes = {root}
        for child in self.get_children(root):
            nodes = nodes.union(self.get_non_leaf_nodes(child))
        return nodes

    def get_subgraph(self, new_root: str):
        """
        @param new_root: Node from which to propagate.
        @return: A new ExpandedActionHierarchy instance containing all nodes and edges reachable from new_root, i.e. a
        new action hierarchy with new_root as the root action.
        """
        actions = {new_root: self.actions[new_root]}
        edges = {}
        subgraph = ExpandedActionHierarchy(actions, edges, self.state_variables, self.primitive_action_map)
        for child in self.get_children(new_root):
            child_subgraph = self.get_subgraph(child)
            subgraph = self.union(subgraph, child_subgraph)
            subgraph.bind_subtask(new_root, child)
        return subgraph

    @staticmethod
    def union(graph_a: 'ExpandedActionHierarchy', graph_b: 'ExpandedActionHierarchy'):
        """
        Combines two ExpandedActionHierarchy instances.

        @param graph_a: First ExpandedActionHierarchy to be merged.
        @param graph_b: Second ExpandedActionHierarchy to be merged.
        @return: ExpandedActionHierarchy instance with the union of the nodes and edges.
        """
        # Make new action hierarchy with union of state variables and primitive action maps
        new_graph = ExpandedActionHierarchy(
            dict(graph_a.actions, **graph_b.actions),
            dict(graph_a.edges, **graph_b.edges),
            graph_a.state_variables.union(graph_b.state_variables),
            dict(graph_a.primitive_action_map, **graph_b.primitive_action_map)
        )

        return new_graph


def _build_graph(source_graph: ParameterizedActionHierarchy, bound_variable_values: Dict[str, Set[str]], root: str,
                 root_old_name: Optional[str] = None, actions: Optional[Dict[str, dict]] = None,
                 edges: Optional[Dict[str, Set[str]]] = None, ground_variable_map: Optional[Dict[str, Any]] = None,
                 bound_var_map: Optional[Dict[str, Any]] = None)\
        -> Tuple[Dict[str, dict], Dict[str, Set[str]], Dict[str, Any]]:
    """
    @param source_graph: ParameterizedActionHierarchy object to be expanded.
    @param bound_variable_values: Dict mapping state variables to their possible values.
    @param root: Root action name.
    @param root_old_name: For recursion, ParameterizedActionHierarchy name of this action.
    @param actions: For recursion, accumulated action dict to be returned.
    @param edges: For recursion, accumulated edges dict to be returned.
    @param ground_variable_map: For recursion, map each parameterized action name to the extra variables and their values
        that are needed for termination predicates and psuedo-rewards. These variables are used for grounded-actions
        For taxi world: Navigate_0_4: {'target': [0,4]}
    @param bound_var_map: Dict of form {'Action Prefix: {'Action suffix' : 'Bound var values'}}
        For navigate, {'Navigate': {'_0_0':(0,0), '_4_4' : (4,4)...}
    @return: Actions and edges for the new ExpandedActionHierarchy, as well as a dictionary which points from each
    parameterized action name to the corresponding value (e.g. {'Navigate_1_2' -> '1_2'})
    """
    # Empty graph
    if actions is None:
        actions = {}
    if edges is None:
        edges = {}
    if root_old_name is None:
        root_old_name = root
    if ground_variable_map is None:
        ground_variable_map = {}

    actions[root] = source_graph.actions[root_old_name]
    edges[root] = set()
    children = source_graph.get_children(root_old_name)
    next_roots = []  # [(new_name, old_name)]

    for child, bound_variables in children:
        # If there are bound variables we need to make duplicates
        if bound_variables:
            # Possible values is union of relevant values in bound_variable_values
            possible_values = set()
            for variable in bound_variable_values.keys():
                if variable in bound_variables:
                    possible_values = possible_values.union(bound_variable_values[variable])
            #print(possible_values)
            # Name for the child node will be "{name}_{value}"
            for value in possible_values:
                new_name = f"{child}_{value}"
                if new_name not in actions.keys():
                    # Add action and info
                    actions[new_name] = source_graph.actions[child]
                    next_roots.append((new_name, child))

                    # create map mapping string to actual values for grounded params
                    try:
                        ga_dict = source_graph.actions[child]['grounded_actions']
                        ground_variable_map[new_name] = {}
                        for ground_var in ga_dict.keys():
                            for variable in bound_variables:
                                if variable in ga_dict[ground_var]:
                                    if bound_var_map:
                                        v = bound_var_map[value]
                                    # Add parameter key
                                    ground_variable_map[new_name][ground_var] = v
                    except KeyError:
                        pass
                # Add edge
                edges[root].add(new_name)
        else:
            # Add node and edge normally
            # Add node if it doesn't already exist
            if child not in actions:
                actions[child] = source_graph.actions[child]
                next_roots.append((child, None))  # None because we're not changing the name
            # Add edge regardless
            edges[root].add(child)

    # Recursive call, propagate down action graph
    for next_root, old_name in next_roots:
        _build_graph(source_graph, bound_variable_values, next_root, old_name, actions, edges, ground_variable_map,
                     bound_var_map=bound_var_map)

    return actions, edges, ground_variable_map
