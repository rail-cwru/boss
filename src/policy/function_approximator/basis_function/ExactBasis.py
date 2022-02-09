from policy.function_approximator.basis_function.base import BasisFunction
import abc
import numpy as np
from functools import reduce

class ExactBasis(BasisFunction):

    """Basis function with no functional approximation.

    This can only be used in domains with finite, discrete state-spaces. For
    example the Chain domain from the LSPI paper would work with this basis,
    but the inverted pendulum domain would not.

    Parameters
    ----------
    num_states: list
        A list containing integers representing the number of possible values
        for each state variable.
    num_actions: int
        Number of possible actions.
    """

    def __init__(self, num_states, num_actions):
        """Initialize ExactBasis."""
        if len(np.where(num_states <= 0)[0]) != 0:
            raise ValueError('num_states value\'s must be > 0')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        self._num_states = num_states

        self._offsets = [1]
        for i in range(1, len(num_states)):
            self._offsets.append(self._offsets[-1]*num_states[i-1])

    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return reduce(lambda x, y: x*y, self._num_states, 1)*self.__num_actions

    def get_state_action_index(self, state, action):
        """Return the non-zero index of the basis.

        Parameters
        ----------
        state: numpy.array
            The state to get the index for.
        action: int
            The action get the index for.

        Returns
        -------
        int
            The non-zero index of the basis

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        """
        if action < 0:
            raise IndexError('action index must be >= 0')
        if action >= self.num_actions:
            print('Action:', action)
            raise IndexError('action must be < num_actions')
        if isinstance(action, list) or isinstance(action, np.ndarray):
            action = action[0]

        base = action * int(self.size() / self.__num_actions)
        offset = 0
        for i, value in enumerate(state):
            offset += self._offsets[i] * state[i]
        return base + offset

    def get_state_action_index_batch(self, state, actions: list):
        """Return the non-zero index of the basis.

        Parameters
        ----------
        state: numpy.array
            The state to get the index for.
        action: int
            The action get the index for.

        Returns
        -------
        int
            The non-zero index of the basis

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        """
        # if action < 0:
        #     raise IndexError('action index must be >= 0')
        # if action >= self.num_actions:
        #     print('Action:', action)
        #     raise IndexError('action must be < num_actions')
        # if isinstance(action, list) or isinstance(action, np.ndarray):
        #     action = action[0]

        base = [i * int(self.size() / self.__num_actions) for i in actions]

        offset = 0
        for i, value in enumerate(state):
            offset += self._offsets[i] * state[i]
        return base + offset

    def evaluate(self, state, action):
        """

        :param state: The state to get the features for. When calculating Q(s, a) this is he s.
        :param action:
        :return:
        """
        r"""Return a :math:`\phi` vector that has a single non-zero value.

        Parameters
        ----------
        state: numpy.array
            
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """
        if len(state) != len(self._num_states):
            raise ValueError('Number of state variables must match '
                             + 'size of num_states.')
        if len(np.where(state < 0)[0]) != 0:
            raise ValueError('state cannot contain negative values.')
        for state_var, num_state_values in zip(state, self._num_states):
            if state_var >= num_state_values:
                raise ValueError('state values must be <= corresponding '
                                 + 'num_states value.')

        phi = np.zeros(self.size())
        phi[self.get_state_action_index(state, action)] = 1

        return phi

    def evaluate2(self, state, action):
        """
        A basis function evaluator that returns the index for optimization
        :param state:
        :param action:
        :return: Returns the basis function, phi and the only non-zero index
        """

        if len(state) != len(self._num_states):
            print(state, self._num_states)
            raise ValueError('Number of state variables must match '
                             + 'size of num_states.')
        # if len(np.where(state < 0)[0]) != 0:
        #     raise ValueError('state cannot contain negative values.')

        # todo: Comment out check for faster runtime
        # for state_var, num_state_values in zip(state, self._num_states):
        #     if state_var >= num_state_values:
        #         raise ValueError('state values must be <= corresponding '
        #                          + 'num_states value.')

        phi = np.zeros(self.size())
        ind = self.get_state_action_index(state, action)
        phi[ind] = 1
        return phi, ind


    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value
