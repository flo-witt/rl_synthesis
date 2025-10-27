import numpy as np

from paynt.quotient.pomdp_family import PomdpFamilyQuotient

import tf_agents
from tf_agents.trajectories import TimeStep

import tensorflow as tf



class FamilyQuotientNumpy:
    """
    A class to contain the family quotient in numpy format.
    """
    def __init__(self, family_quotient: PomdpFamilyQuotient):
        """
        Initializes the FamilyQuotientNumpy with a PomdpFamilyQuotient instance.

        :param family_quotient: An instance of PomdpFamilyQuotient.
        """
        self.family_quotient = family_quotient
        self._initialize_numpy_arrays(family_quotient)
        self._compute_observation_mapping_function(family_quotient.obs_evaluator)

    def _compute_observation_mapping_function(self, obs_evaluator):
        valuations_length = len(obs_evaluator.obs_valuation(0))
        self.observation_mapping_function = np.zeros((self.family_quotient.num_observations, valuations_length), dtype=np.float32)
        for observation_integer in range(self.family_quotient.num_observations):
            valuation = obs_evaluator.obs_valuation(observation_integer)
            if len(valuation) != valuations_length:
                raise ValueError(f"Observation valuation length mismatch: expected {valuations_length}, got {len(valuation)} for observation {observation_integer}")
            self.observation_mapping_function[observation_integer] = np.array(valuation, dtype=np.float32)

    def _initialize_numpy_arrays(self, family_quotient: PomdpFamilyQuotient):
        """
        Initializes numpy arrays for the family quotient.
        """
        self.action_labels = np.array(family_quotient.action_labels, dtype=str)
        observation_to_actions = family_quotient.observation_to_actions # array of shape (num_observations, None), where None is the variable length of actions for each observation
        self.observation_to_legal_action_mask = np.zeros(
            (family_quotient.num_observations, len(self.action_labels)),
            dtype=bool
        )
        for i, observation_to_actions in enumerate(observation_to_actions):
            self.observation_to_legal_action_mask[i, observation_to_actions] = True
        
    def get_time_steps_for_observation_integers(self, observation_integer: int | list[int], new_action_labels : list[str]) -> tuple[TimeStep, np.ndarray[bool]]:
        """
        Returns the time steps for the given observation integers.

        :param observation_integers: A tensor of observation integers.
        :return: A TimeStep object containing the time steps for the given observations.
        """
        if type(observation_integer) is int:
            observation_integer = [observation_integer]

        # Create a convertor from original action space mapping to new action space given the original and new action labels
        action_mapping = np.array([new_action_labels.index(label) if label in new_action_labels else len(new_action_labels) for label in self.action_labels])

        np_observation_integers = np.array(observation_integer, dtype=np.int32)
        legal_actions = self.observation_to_legal_action_mask[np_observation_integers]
        mapped_legal_actions = np.zeros((len(np_observation_integers), len(new_action_labels) + 1), dtype=bool)
        mapped_legal_actions[:, action_mapping] = legal_actions
        illegal_actions_flags = mapped_legal_actions[:, -1]
        mapped_legal_actions = mapped_legal_actions[:, :-1]  # Remove the last column which is the illegal actions flag
        observation_valuations = self.observation_mapping_function[np_observation_integers]

        time_step = TimeStep(
            step_type=tf.constant([tf_agents.trajectories.StepType.MID] * len(np_observation_integers), dtype=tf.int32),
            reward=tf.constant([0.0] * len(np_observation_integers), dtype=tf.float32),
            discount=tf.constant([1.0] * len(np_observation_integers), dtype=tf.float32),
            observation={
                "integer": tf.constant(np_observation_integers.reshape(-1, 1), dtype=tf.int32),
                "mask": tf.constant(mapped_legal_actions, dtype=tf.bool),
                "observation": tf.constant(observation_valuations, dtype=tf.float32)
            }
        )

        return time_step, illegal_actions_flags