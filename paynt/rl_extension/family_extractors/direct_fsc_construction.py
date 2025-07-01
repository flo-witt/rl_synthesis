
import tensorflow as tf

from rl_src.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec

from paynt.quotient.fsc import FSC
from paynt.quotient.pomdp import PomdpQuotient
from paynt.rl_extension.robust_rl.family_quotient_numpy import FamilyQuotientNumpy

class ConstructorFSC:
    """Class to construct a Finite State Controller (FSC) from different policies, e.g. TableBasedPolicy.
    This class provides a method to convert a table-based policy into an FSC, which can be used for reinforcement learning tasks.
    """

    @staticmethod
    def __create_action_function(tf_action_function : tf.Tensor, family_quotient_numpy: FamilyQuotientNumpy = None, original_action_labels: list[str] = None):
        """Creates the action function for the FSC.
        Args:
            tf_action_function (tf.Tensor): The action function to be used in the FSC.
        Returns:
            tf.Tensor: The created action function.
        """
        np_action_function = tf.cast(tf_action_function, dtype=tf.float32).numpy()
        if np_action_function.shape[-1] == 1:
            return np_action_function.astype(int).tolist()  # Deterministic action function
        else:
            # Convert the probabilites at the lowest level to dictionaries of selected action and its probability
            action_function = []
            for memory in range(np_action_function.shape[0]):
                action_for_memory = []
                for observation in range(np_action_function.shape[1]):
                    action_dict = {}
                    for action, prob in enumerate(np_action_function[memory][observation]):
                        if prob > 0.0:
                            action = int(action) if (family_quotient_numpy and original_action_labels) is None else family_quotient_numpy.action_labels.tolist().index(original_action_labels[action])
                            action_dict[action] = prob
                    if action_dict == {}:
                        if family_quotient_numpy is not None:
                            action_dict = {action: 1.0 for action in range(len(family_quotient_numpy.action_labels)) if family_quotient_numpy.observation_to_legal_action_mask[observation][action]}
                            action_dict = {action: 1.0 / len(action_dict) for action in action_dict}  # Uniform distribution over legal actions
                    action_for_memory.append(action_dict)
                action_function.append(action_for_memory)
            return action_function
        
    @staticmethod
    def __create_update_function(tf_update_function : tf.Tensor):
        """Creates the update function for the FSC.
        Args:
            tf_update_function (tf.Tensor): The update function to be used in the FSC.
        Returns:
            tf.Tensor: The created update function.
        """

        np_update_function = tf.cast(tf_update_function, dtype=tf.float32).numpy()
        if np_update_function.shape[-1] == 1: # Deterministic update
            # If there is only one update function, return it as a list
            return np_update_function.astype(int).tolist()  # Deterministic update function
        else:
            # Convert the probabilites at the lowest level to dictionaries of selected update and its probability
            update_function = []
            for memory in range(np_update_function.shape[0]):
                update_for_memory = []
                for observation in range(np_update_function.shape[1]):
                    update_dict = {}
                    for update, prob in enumerate(np_update_function[memory][observation]):
                        if prob > 0:
                            update_dict[update] = prob
                    update_for_memory.append(update_dict)
                update_function.append(update_for_memory)
            return update_function

    @staticmethod
    def __create_observation_labels(pomdp_quotient : PomdpQuotient):
        return pomdp_quotient.observation_labels
    
    @staticmethod
    def __create_action_labels(table_based_policy : TableBasedPolicy):
        return list(table_based_policy.action_keywords)

    @staticmethod
    def construct_fsc_from_table_based_policy(
        table_based_policy: TableBasedPolicy,
        pomdp_quotient: PomdpQuotient,
        family_quotient_numpy: FamilyQuotientNumpy,

        ) -> FSC:
        """Constructs a Finite State Controller (FSC) from a table-based policy.
        Args:
            table_based_policy (TableBasedPolicy): The table-based policy to be converted into an FSC.
            environment_wrapper (EnvironmentWrapperVec): The environment wrapper used to create the FSC.
        Returns:
            FSC: The constructed FSC.
        """
        # Create a new FSC object
        action_function = ConstructorFSC.__create_action_function(table_based_policy.tf_observation_to_action_table, family_quotient_numpy, table_based_policy.action_keywords)
        update_function = ConstructorFSC.__create_update_function(table_based_policy.tf_observation_to_update_table)
        is_deterministic = True if type(action_function[0][0]) is int else False
        try:
            observation_labels = ConstructorFSC.__create_observation_labels(pomdp_quotient)
        except:
            observation_labels = None
        action_labels = ConstructorFSC.__create_action_labels(table_based_policy) if family_quotient_numpy is None else family_quotient_numpy.action_labels.tolist()

        num_observations = len(action_function[0])
        num_nodes = len(action_function)

        # Create the FSC
        fsc = FSC(
            action_function=action_function,
            update_function=update_function,
            observation_labels=observation_labels,
            action_labels=action_labels,
            is_deterministic=is_deterministic,
            num_observations=num_observations,
            num_nodes=num_nodes,
        )
        
        return fsc