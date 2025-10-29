from tools.args_emulator import ArgsEmulator
import tensorflow as tf

import numpy as np

class BinaryShield:
    def __init__(self, nr_actions: int, args : ArgsEmulator = None):
        self.args = args
        self.nr_actions = nr_actions
        # TODO: Add the temporary shielding stuff.
        self.shielding_method = self.dummy_shield

    @staticmethod  
    def dummy_shield(state, prev_action, step_types, nr_actions):
        return [True] * nr_actions

    def compute_new_mask(self, valuations : list, integers : list, actions : list, resets : list) -> np.ndarray[np.bool_]:
        """ A dummy shielding method that always allows the action.
        Args:
            valuations: The valuations of a current environment state/observation.
            integers: The integer representation of the current environment state/observation.
            action: The previous action taken by the agent.
            resets: Whether the episode is a restarted simulation in a current state ([True/False]).
        Returns:
            bool: True if the action is allowed, False otherwise.
        """

        masks = []
        for i in range(len(valuations)):
            distribution = self.shielding_method(integers[i], actions[i], resets[i], self.nr_actions)
            # If the distribution is larger than 0, True, otherwise False.
            mask = [val > 0.0 for val in distribution]
            masks.append(mask)
        return np.array(masks, dtype=np.bool_)
        
