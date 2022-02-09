"""
Common class for common Domain Transfer operations and classes, etc.
"""
from typing import Union, List, Tuple, Dict, Any
import copy


class DomainTransferMessage(object):
    """
    Message describing a domain transfer.
    Should be created when performing a domain transfer.

    In general, attempt to make sure that illegal domain transfer messages are entirely not created
        rather than checking legality once the message has been created.
    """

    def __init__(self,
                 __sentinel=None,
                 remap_agents: Dict[int, Union[int, type(None)]]=None,
                 add_agent_class_id_map: Dict[int, int]=None,
                 # del_agent_class_ids: Union[int, List[int], Tuple[int, ...]]=None,
                 # add_agent_class_ids: Union[int, List[int], Tuple[int, ...]]=None,
                 # altered_agent_class_obs_domains: Union[int, List[int], Tuple[int, ...]]=None,
                 # altered_agent_class_act_domains: Union[int, List[int], Tuple[int, ...]]=None,
                 data: Any=None
                 ):
        """
        Create a Domain Transfer Message.
        :param __sentinel: USED TO MAKING SURE ALL ARGUMENTS ARE NAMED
        :param remap_agents: A dictionary mapping from IDs to modify. This is used to remap or delete agents.
            WARNINGS: Only delete existing agents and do NOT remap between different agent-classes.
            Map an ID to another ID to indicate remapping. This overwrites the target agent.
            {A_TO: A_FROM} meaning that agent of id A_TO is replaced with agent with id A_FROM.
            A_FROM is removed from is previous position.
                In the case that an agent is overwritten without being mapped elsewhere, it is identical to deletion.
                    e.g. {5: 3} means that agent 3 will be replaced with agent 5.
                    {3: 5}       : [1, 2, 3, 4, 5] -> [1, 2, 5, 4]      deleted 3
                        If same in both env and asys, it is deletion of agent 3.
                        Identical in function to {4: 3}, but just that the organization of data is different.
                    {1: 2, 2: 1} : [1. 2. 3. 4. 5] -> [2, 1, 3, 4, 5]
                Map an agent to None to delete the agent, moving the agent listing all down one step.
                It is illegal for agents to be mapped to nonexistent canonical IDs. An error will be thrown.
        :param add_agent_class_id_map: IDs to insert. Map of agent class (within which to insert new agent)
            to number of new agents to insert. New agent IDs will be automatically placed at highest index.
        :forbidden del_agent_class_ids: Deleted agent classes.
        :forbidden add_agent_class_ids: Added agent classes.
            Added agent class will be inserted at the end.
        :forbidden altered_agent_class_obs_domains:
            IDs of agent classes which observation domain is altered.
            This is not allowed to change in an experiment.
        :forbidden altered_agent_class_act_domains:
            IDs of agent classes which action domain is altered.
            This is not allowed to change in an experiment.
        :param data: Any sort of data. Might be used by environment or agentsystem to effect special behavior.

        remap_agents, if different between the env and asys, can allow for interesting things to occur:
        1:  env : {3: 5   } : [1, 2, 3, 4, 5] -> [1, 2, 5, 4]
            asys: {3: 5   } : [1, 2, 3, 4, 5] -> [1, 2, 5, 4]
            result: Effective deletion of agent 3, policies remain same.
        2:  env : {3: 5   } : [1, 2, 3, 4, 5] -> [1, 2, 5, 4]
            asys: {5: None} : [1, 2, 3, 4, 5] -> [1, 2, 3, 4]
            result: 5 in env takes the policy of 3 in asys.
        2:  env : {1: 3   } : [1, 2, 3, 4, 5] -> [3, 2, 4, 5]
            asys: {3: None} : [1, 2, 3, 4, 5] -> [1, 2, 4, 5]
            result: 3 in env takes the policy of 1 in asys.
        """
        # Remap agent map. (Make deletions explicit)
        full_remap = {}
        for a_write, a_read in remap_agents.items():
            full_remap[a_write] = a_read
            if a_read not in full_remap and a_read is not None:
                full_remap[a_read] = None
        self.agent_remapping = full_remap

        self.add_agent_class_id_map = add_agent_class_id_map
        # self.altered_agent_class_obs_domains = altered_agent_class_obs_domains
        # self.altered_agent_class_act_domains = altered_agent_class_act_domains
        self.data = data

    def remap_id_list(self, ids):
        mapped_ids: List = copy.deepcopy(ids)
        for a_write, a_read in self.agent_remapping.items():
            mapped_ids[ids.index(a_write)] = a_read
        mapped_ids = [a_id for a_id in mapped_ids if a_id is not None]
        return mapped_ids

