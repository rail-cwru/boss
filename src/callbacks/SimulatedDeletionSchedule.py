from typing import TYPE_CHECKING, List

import numpy as np

from agentsystem import CoordinationGraphSystem
from callbacks.base import ScheduleCallback, CallbackImpl
from common.domain_transfer import DomainTransferMessage
from common.properties import Properties
from config.config import ConfigItemDesc
from config import gen_check, checks

if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class SimulatedDeletionSchedule(ScheduleCallback):
    """
    Compute deletion sensitivity by simulating episodes under deletion before actually deleting agents at the beginning
        of certain episodes.

    Measures degradation of performance by decrease in mean cumulative reward for all agents.
    """

    # TODO we want to test deletion during episode in future.

    @classmethod
    def properties(cls):
        return Properties(use_agent_deletion=True)

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='schedule',
                           # Can only check at init time...
                           check=gen_check.schedule_of(lambda l: isinstance(l, list)),
                           info='Dictionary schedule of list of agents to delete at various episode times.'),
            ConfigItemDesc(name='eval_num',
                           check=checks.positive_integer,
                           info='Number of times to simulate deletion of each agent'),
            ConfigItemDesc(name='eval_at',
                           check=checks.positive_integer,
                           info='Episode at which to compute agent-fixed deletion sensitivity.'),
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        self.eval_num = self.callback_config.eval_num
        self.eval_at = self.callback_config.eval_at
        assert self.eval_at <= sorted(self.schedule.keys())[0], 'Must evaluate deletion sensitivity before deletion.'
        for k, v in self.schedule.items():
            # TODO (future/algorithmic) simulated single deletion is even less accurate if deleting multiple agents.
            assert len(v) == 1, 'Simulated Deletion does not support simultaneous deletion of more than one agent.'
            # Trust user to not sequentially delete all agents or result in invalid deletion (e.g. repeatedly del last)
            for agent in v:
                err_msg = 'Scheduled deletion of nonexistent agent [{}] before episode [{}]'.format(k, agent)
                assert agent in self.controller.env.agent_id_list, err_msg
        self.deletion_trials = {}
        self.agent_mean_reward_after_deletion = {}
        self.deletion_order = []

    def _get_implement_flags(self) -> CallbackImpl:
        return CallbackImpl(before_episode=True, finalize=True)

    def before_episode(self):
        if self.controller.episode_num == self.eval_at:
            deletion_info = []
            checkpoint = self.controller.get_checkpoint()
            self.controller.flags.exploit = True
            self.controller.flags.learn = False
            # Simulate deletion of each agent N times.
            agents = self.controller.env.agent_id_list
            for agent in agents:
                self.controller.set_from_checkpoint(checkpoint)
                # deleting agent second time with high deletion probability outputs ? what? issue TODO BUG
                self.controller.env.transfer_domain(DomainTransferMessage(remap_agents={agent: None}))
                mean_rewards = []
                # Run N episodes with each agent deleted and obtain the episode-mean of the agent-mean rewards.
                for _ in range(self.eval_num):
                    total_reward_dict = self.controller.run_episode().get_agent_total_rewards()
                    mean_rewards.append(np.mean([reward for reward in total_reward_dict.values()]))
                mean_reward_after_deletion = np.mean(mean_rewards)
                self.agent_mean_reward_after_deletion[agent] = mean_reward_after_deletion
                self.deletion_trials[agent] = mean_rewards
                print('Agent {} deletion trials: '.format(agent) + str(mean_rewards))
                deletion_info.append([agent, mean_reward_after_deletion])
            self.deletion_order = sorted(deletion_info, key=lambda duple: -duple[1])
            self.controller.set_from_checkpoint(checkpoint)
            print('Deletion sensitivity: ', self.agent_mean_reward_after_deletion)
        super(SimulatedDeletionSchedule, self).before_episode()

    def on_trigger(self, value):
        # Let A_SENS = agent with least sensitivity
        # Let A_DEL  = agent to be deleted from environment
        # TODO (future / algorithmic) - simultaneous deletion
        a_sens = self.deletion_order[0][0]
        a_del = value[0]
        deletion_msg = DomainTransferMessage(remap_agents={a_del: None})
        proxy_deletion_msg = DomainTransferMessage(remap_agents={a_del: a_sens})
        # env  remap: {A_DEL  <- None }
        self.controller.env.transfer_domain(deletion_msg)

        # Don't let the controller catch the message we don't want ASYS to receive.
        self.controller.env.pop_last_domain_transfer()

        # asys remap: {A_DEL <- A_SENS}
        # asys: CoordinationGraphSystem = self.controller.asys
        # print('Asys Before Deletion:', asys.graph)
        if a_del != a_sens:
            # print('Deleting {} instead of {}'.format(a_sens, a_del))
            self.controller.asys = self.controller.asys.transfer_domain(proxy_deletion_msg)
        else:
            # print('Deleting {} normally'.format(a_del))
            self.controller.asys = self.controller.asys.transfer_domain(deletion_msg)
        # print('Asys After Deletion:', asys.graph)

    def finalize(self):
        return {'deletion_trials': self.deletion_trials, 'deletion_sensitivity': self.agent_mean_reward_after_deletion}
