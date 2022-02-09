# Verify the action selection techniques.
from common.domain_transfer import DomainTransferMessage
from controller.mdp_controller import MDPController
from config import Config


def test_cgql_learning():
    config = Config('../../experiment_configs/debug_cgql.json')
    controller = MDPController(config)
    controller.run()


def test_cgql_actionselection():
    config = Config('../../experiment_configs/debug_cgql.json')
    # Tests PredatorPrey
    # Tests DiscreteActionFeatureDomain
    # Tests DiscreteIndexedActionDomain
    # Tests Domain Joining
    # Tests ActionFreeLinear
    # This will be much nicer once ConfigItem is a thing.
    for method_to_test in ['greedy_with_restarts', 'max_plus', 'agent_elimination']:
        config.agentsystem.selection_method = method_to_test
        controller = MDPController(config)
        observation_request = controller.asys.observe_request()
        agents_observation = controller.env.observe(observation_request)
        agent_action_map = controller.asys.get_actions(agents_observation)

        # For one-edge, two-agent system
        # action_map = self.action_selection(all_action_q)
        # for agent_id in self.agent_ids:
        #     if (self.get_total_q(action_map, all_action_q) != all_action_q[0].max()).all():
        #         print('DISAGREEMENT')
        #         action_map = self.action_selection(all_action_q)

        # FOR VISUALIZING JOINT DECISION MAKING FOR SINGLE EDGES
        # joint_action = joint_act_dom.extract_sub_actions(np.array([np.argmax(all_action_q[edge_id])]))
        # preferred = {aid: act for aid, act in zip(self.graph.edges[edge_id], joint_action)}
        # import matplotlib.pyplot as plt
        # if not hasattr(self, 'fig'):
        #     self.fig, self.ax = plt.subplots(ncols=2, figsize=(6, 3))
        #     self._im0 = self.ax[0].imshow(edge_observation.reshape(joint_act_dom.full_range, -1))
        #     self._cb0 = self.fig.colorbar(self._im0, ax=self.ax[0])
        #     self._im1 = self.ax[1].imshow(all_action_q[edge_id])
        #     self._cb1 = self.fig.colorbar(self._im1, ax=self.ax[1])
        # else:
        #     self._im0.set_data(edge_observation.reshape(joint_act_dom.full_range, -1))
        #     self._cb0.set_clim(edge_observation.min(), edge_observation.max())
        #     self._im1.set_data(all_action_q[edge_id])
        #     self._cb1.set_clim(all_action_q[edge_id].min(), all_action_q[edge_id].max())
        # plt.ion()
        # # return preferred
        # END DEBUG


    print("Tests completed successfully!")


def test_cgql_deletion():
    config = Config('../../experiment_configs/debug_cgql.json')
    config.callbacks = {}
    config.episodes = 1
    for transfer_method in ['drop_removed', 'map_to_neighbors', 'project_to_existing']:
        config.agentsystem.transfer_method = transfer_method
        controller = MDPController(config)
        controller.run()
        dtm = DomainTransferMessage(remap_agents=[0])
        transferred = controller.asys.transfer_domain(dtm)
    print('CoordinationGraphSystem is able to run domain transfer.')


if __name__ == '__main__':
    test_cgql_deletion()
