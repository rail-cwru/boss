from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, TYPE_CHECKING, List

from callbacks import CallbackImpl
from common import Properties
from config.config import ConfigItemDesc
from policy import Policy
from config import checks
from . import Callback
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA

if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class VisualizeFA(Callback):
    """
    Visualizes Function-Approximator after [timestep] episodes or at every step.
    For weight-based FA, visualizes weights.
    For tabular FA, visualizes table values.

    Will try to fit weights on the screen.
    """

    @classmethod
    def properties(cls):
        return Properties(use_function_approximator=True)

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='timestep',
                           check=checks.positive_integer,
                           info='Interval of episodes after which to run this callback.'),
            ConfigItemDesc(name='all_steps',
                           check=lambda b: isinstance(b, bool) or checks.positive_integer(b),
                           info='Whether or not to visualize FA at every step in active episodes.\n'
                                'If bool, visualizes every step. If int, visualizes every n steps.'),
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        # Map iteration number to a list of trajectories evaluated at that iteration.
        callback_config = config.callbacks['VisualizeFA']
        # Number of episodes between each eval
        self.eval_timestep = callback_config.timestep

        self.all_steps = int(callback_config.all_steps)
        self.do_after_update = self.all_steps > 0

        self.unique_policies_weights = OrderedDict()        # Maps unique policies to FA weight dicts.
        self.policy_pg_map = {}                             # Maps unique policies to PG IDs.
        self.instanced_plots = {}                           # Maps policy and weightname to plot for that weight
        self.fig: plt.Figure = None                         # Figure
        self.ax: np.ndarray = None                          # Axes
        self.fig_active = False                             # Whether or not the figure is active
        err_msg = 'VisualizeFA Requires use of FA Policies.'
        assert issubclass(config.policy.module_class, Policy), err_msg
        super().__init__(controller, config)

    def _get_implement_flags(self):
        return CallbackImpl(before_run=True, after_update=self.do_after_update, after_episode=True)

    def before_run(self):
        self.__extract_weights()

        # Columns are each weight-array, rows are each unique policy. Figure out which is longer.
        cols = max([len(weight_dict) for weight_dict in self.unique_policies_weights.values()])
        rows = len(self.unique_policies_weights)

        # Setup plot function
        self.fig, self.ax = plt.subplots(nrows=rows, ncols=cols)
        if rows * cols == 1:
            self.ax = np.array([[self.ax]])
        elif rows == 1 or cols == 1:
            self.ax = self.ax.reshape(cols, rows)
        plt.ion()
        self.fig.show()

        self.fig_active = True

    def after_update(self):
        episode_num = self.controller.episode_num
        if episode_num % self.eval_timestep == 0 or episode_num == (self.controller.episodes - 1) \
                and self.fig_active and (self.controller.episode_step % self.all_steps) == 0:
            self.__extract_weights()
            self.__plot_weights()

    def after_episode(self):
        episode_num = self.controller.episode_num
        if episode_num % self.eval_timestep == 0 or episode_num == (self.controller.episodes - 1) and self.fig_active:
            self.__extract_weights()
            self.__plot_weights()

    def __plot_weights(self):
        # TODO It'd be nice to plot the feature names for the FA's weights where applicable...
        for i_pol, pair, in enumerate(self.unique_policies_weights.items()):
            policy, weight_dict = pair
            pg_ids = self.policy_pg_map[policy]
            col_header = 'Policy for Policy Group{} {}'.format('s' if len(pg_ids) > 0 else '', pg_ids[0])
            for pgid in pg_ids[1:]:
                col_header += ', {}'.format(pgid)
            for i_weight, weight_item in enumerate(weight_dict.items()):

                title, weight_arr = weight_item

                if not isinstance(weight_arr, np.ndarray):
                    # TODO this is a bad hacky workaround
                    continue

                ax: plt.Axes = self.ax[i_weight, i_pol]
                plot_key = (policy, title)

                weight_arr = np.squeeze(weight_arr)

                # Detect changes in structure
                if plot_key in self.instanced_plots:
                    # Update existing plot
                    plot = self.instanced_plots[plot_key]
                    if weight_arr.ndim == 1:
                        lim = max(abs(weight_arr)) * 1.05
                        ax.set_xlim(-lim, lim)
                        for rect, x in zip(plot, weight_arr):
                            rect.set_width(x)
                    else:
                        plot[0].set_data(weight_arr)
                        w_min = weight_arr.min()
                        w_max = weight_arr.max()
                        plot[1].set_clim(w_min, w_max)
                        cbar_ticks = np.linspace(w_min, w_max+1e-9, num=6, endpoint=True)
                        plot[1].set_ticks(cbar_ticks)
                        plot[1].draw_all()
                else:
                    if i_weight == 0:
                        title = col_header + '\n' + title
                    self.instanced_plots[plot_key] = self.__create_subplot(ax, title, weight_arr)
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            self.fig_active = False

    def __create_subplot(self, ax, title, weight_arr):
        """
        Newly creates a subplot for data.
        :param ax: Axis to plot to
        :param title: Title for the subplot
        :param weight_arr: Data to plot
        :return: Plot object
        """
        # Try and get feats to be vertical, acts to be horizontal
        ax.set_title(title)
        if weight_arr.ndim == 1:
            lim = max(abs(weight_arr)) * 1.05
            ax.invert_yaxis()
            ax.set_xlim(-lim, lim)
            return ax.barh(np.r_[:len(weight_arr)], weight_arr)
        elif weight_arr.ndim == 2:
            im = ax.imshow(weight_arr)
            cb = self.fig.colorbar(im, ax=ax)
            return im, cb
        elif weight_arr.ndim > 2:
            ravel = weight_arr.ravel()
            dim_2d = np.ceil(np.sqrt(ravel.size))
            padded_square = np.square(dim_2d)
            ravel = np.append(ravel, np.zeros(padded_square - ravel.size, dtype=ravel.dtype))
            im = ax.imshow(ravel.reshape(dim_2d, dim_2d))
            cb = self.fig.colorbar(im, ax=ax)
            w_min = ravel.min()
            w_max = ravel.max()
            cbar_ticks = np.linspace(w_min, w_max + 1e-9, num=6, endpoint=True)
            cb.set_ticks(cbar_ticks)
            return im, cb

    def __extract_weights(self):
        """
        Extract weight dicts to self.unique_policies_weights and regenerates self.policy_pg_map.
        """
        self.unique_policies_weights = OrderedDict()
        self.policy_pg_map = {}
        for pg in self.controller.asys.policy_groups:
            policy: Policy = pg.policy
            if policy not in self.unique_policies_weights:
                if isinstance(policy.function_approximator, AbstractPyTorchFA):
                    self.unique_policies_weights[policy] = policy.function_approximator.get_variable_vals()['model']
                else:
                    self.unique_policies_weights[policy] = policy.function_approximator.get_variable_vals()
            if policy not in self.policy_pg_map:
                self.policy_pg_map[policy] = [pg.pg_id]
            else:
                self.policy_pg_map[policy].append(pg.pg_id)

