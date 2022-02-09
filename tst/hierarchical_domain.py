import unittest

from domain.hierarchical_domain import action_hierarchy_from_config, HierarchicalActionDomain

d = {
    "state_variables": [
        "taxi_loc",
        "source",
        "destination",
        "holding_passenger"
    ],
    "actions": {
        "Root": {
            "primitive": False,
            "state_variables": ["taxi_loc", "source", "destination", "holding_passenger"],
            "params": {},
            "parents": [],
            "children": {
                "Get": [],
                "Put": []
            },
            "termination": [[
                ["taxi_loc", "EQUALS", "destination"],
                ["holding_passenger", "EQUALS", "FALSE"]
            ]],
            "psuedo_r": [
                []
            ]
        },
        "Get": {
            "primitive": False,
            "state_variables": ["taxi_loc", "source", "holding_passenger"],
            "params": {},
            "parents": ["Root"],
            "children": {
                "Pickup": [],
                "Navigate": ["source"]
            },
            "termination": [[
                ["holding_passenger", "EQUALS", "TRUE"]
            ]],
            "psuedo_r": [
                ["taxi_loc", "EQUALS", "source", 10]
            ]
        },
        "Put": {
            "primitive": False,
            "state_variables": ["taxi_loc", "destination", "holding_passenger"],
            "params": {},
            "parents": ["Root"],
            "children": {
                "Putdown": [],
                "Navigate": ["destination"]
            },
            "termination": [[
                ["holding_passenger", "EQUALS", "FALSE"]
            ]],
            "psuedo_r": [
                ["taxi_loc", "EQUALS", "destination", 10]
            ]
        },
        "Navigate": {
            "primitive": False,
            "state_variables": ["taxi_loc"],
            "params": {
                "target": ["source", "destination"]
            },
            "parents": ["Get", "Put"],
            "children": {
                "North": [],
                "South": [],
                "East": [],
                "West": []
            },
            "termination": [[
                ["taxi_loc", "EQUALS", "target"]
            ]],
            "psuedo_r": [
                []
            ]
        },
        "Pickup": {
            "primitive": True,
            "parents": ["Get"]
        },
        "Putdown": {
            "primitive": True,
            "parents": ["Put"]
        },
        "North": {
            "primitive": True,
            "parents": ["Navigate"]
        },
        "South": {
            "primitive": True,
            "parents": ["Navigate"]
        },
        "East": {
            "primitive": True,
            "parents": ["Navigate"]
        },
        "West": {
            "primitive": True,
            "parents": ["Navigate"]
        }
    },
    "primitive_action_map": {
        "Pickup": 4,
        "Putdown": 5,
        "North": 0,
        "South": 1,
        "East": 2,
        "West": 3
    }
}

possible_values = {
    'source': {'R', 'G', 'B', 'Y'},
    'destination': {'R', 'B'}
}


class TestHierarchicalDomain(unittest.TestCase):
    def setUp(self) -> None:
        self.action_hierarchy = action_hierarchy_from_config(d)

    def test_graph_was_constructed_correctly(self):
        edges = {
            ('Root', 'Get'): [],
            ('Root', 'Put'): [],
            ('Get', 'Pickup'): [],
            ('Get', 'Navigate'): ['source'],
            ('Put', 'Putdown'): [],
            ('Put', 'Navigate'): ['destination'],
            ('Navigate', 'North'): [],
            ('Navigate', 'South'): [],
            ('Navigate', 'East'): [],
            ('Navigate', 'West'): []
        }
        self.assertEqual(edges, self.action_hierarchy.edges)

    def test_expanded(self):
        edges = {
            'Root': {'Get', 'Put'},
            'Get': {'Navigate_R', 'Navigate_G', 'Navigate_B', 'Navigate_Y', 'Pickup'},
            'Put': {'Navigate_R', 'Navigate_B', 'Putdown'},
            'Pickup': set(),
            'Putdown': set(),
            'Navigate_R': {'East', 'South', 'West', 'North'},
            'Navigate_Y': {'East', 'South', 'West', 'North'},
            'Navigate_G': {'East', 'South', 'West', 'North'},
            'Navigate_B': {'East', 'South', 'West', 'North'},
            'North': set(),
            'South': set(),
            'East': set(),
            'West': set()
        }
        for k in edges.keys():
            self.assertEqual(set(edges[k]), set(self.action_hierarchy.compile(possible_values).edges[k]))

    def test_domain(self):
        expanded = self.action_hierarchy.compile(possible_values)
        h_domain = HierarchicalActionDomain('name', expanded)
        self.assertEqual(6, h_domain.items.pop().range.stop)
        # Test each domain
        root_domain = h_domain.domain_for_action('Root')
        self.assertEqual(2, root_domain.items.pop().range.stop)  # Get, Put
        get_domain = h_domain.domain_for_action('Get')
        self.assertEqual(5, get_domain.items.pop().range.stop)  # Pickup, Navigate R, G, B, Y
        put_domain = h_domain.domain_for_action('Put')
        self.assertEqual(3, put_domain.items.pop().range.stop)  # Putdown, Navigate R, B
        nav_r_domain = h_domain.domain_for_action('Navigate_R')
        self.assertEqual(4, nav_r_domain.items.pop().range.stop)  # N, S, E, W
        nav_g_domain = h_domain.domain_for_action('Navigate_G')
        self.assertEqual(4, nav_g_domain.items.pop().range.stop)  # N, S, E, W
        nav_b_domain = h_domain.domain_for_action('Navigate_B')
        self.assertEqual(4, nav_b_domain.items.pop().range.stop)  # N, S, E, W
        nav_y_domain = h_domain.domain_for_action('Navigate_Y')
        self.assertEqual(4, nav_y_domain.items.pop().range.stop)  # N, S, E, W
