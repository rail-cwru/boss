from typing import TYPE_CHECKING, List

from callbacks.base import ScheduleCallback
from common.domain_transfer import DomainTransferMessage
from config.config import ConfigItemDesc
from config import gen_check

if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class DeleteAgentsSchedule(ScheduleCallback):
    """
    Load an agentsystem's learned policies before certain episodes.

    Loaded policy must be compatible.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='schedule',
                           # Can only check at init time...
                           check=gen_check.schedule_of(lambda l: isinstance(l, list)),
                           info='Dictionary schedule of list of agents to delete at various episode times.')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        for k, v in self.schedule.items():
            for agent in v:
                err_msg = 'Scheduled deletion of nonexistent agent [{}] before episode [{}]'.format(k, agent)
                assert agent in self.controller.env.agent_id_list, err_msg

    def on_trigger(self, value: List[int]):
        message = DomainTransferMessage(remap_agents={v: None for v in value})
        self.controller.env.transfer_domain(message)
