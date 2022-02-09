# Verify the policy sharing of nsymmetricsystem.
from agentsystem.NSymmetricSystem import NSymmetricSystem
from controller.mdp_controller import MDPController
from config import Config


def test_graph_partition():
    config = Config('../../experiment_configs/debug_nsym1.json')

    config.agentsystem.coordination_graph = [[0,1], [1,2], [2,3], [0, 3]]
    controller = MDPController(config)
    asys: NSymmetricSystem = controller.asys
    exemplar = asys.policy_groups[0].policy
    for pg in asys.policy_groups[1:]:
        assert exemplar == pg.policy
    assert asys.shared_policy_map[('predator', 'predator', 0, 0)] == exemplar
    assert asys.partition_map == {0:0, 1:0, 3:0, 2:0}

    # Zebra-striped ring. [01 10 01 10]
    config.agentsystem.degree = 2
    config.agentsystem.coordination_graph = [[0, 1], [1, 2], [2, 3], [0, 3]]
    controller = MDPController(config)
    asys: NSymmetricSystem = controller.asys
    exemplar = asys.policy_groups[0].policy
    for pg in asys.policy_groups[1:]:
        assert exemplar == pg.policy
    assert asys.shared_policy_map[('predator', 'predator', 0, 1)] == exemplar
    assert asys.partition_map == {0: 0, 1: 1, 3: 1, 2: 0}


if __name__ == '__main__':
    test_graph_partition()
