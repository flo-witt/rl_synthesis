from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec
import numpy as np

from interpreters.extracted_fsc.extracted_fsc_policy import ExtractedFSCPolicy

from tests.general_test_tools import init_args, init_environment
from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from tools.evaluators import evaluate_policy_in_model

from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

class TableBasedPolicy(TFPolicy):
    def __init__(self, original_policy : TFPolicy, action_function : np.ndarray, 
                 update_function : np.ndarray, # Action and update function has shape (nr_model_states, nr_observations) 
                 initial_memory = 0,
                 action_keywords = None,
                 descending_actions = None,
                 nr_observations = None
                 ):
        """
        TableBasedPolicy is a policy that uses a table to map observations to actions and updates.
        
        Args:
            original_policy (TFPolicy): The original policy to be used. (TODO: Remove this, since you can derive the policy from the model.)
            action_function (np.ndarray): The action function table. It has shape (nr_model_states, nr_observations).
            update_function (np.ndarray): The update function table. It has shape (nr_model_states, nr_observations)."""
        policy_state_spec = TensorSpec(shape=(), dtype=tf.int32)
        super(TableBasedPolicy, self).__init__(original_policy.time_step_spec, original_policy.action_spec, policy_state_spec=policy_state_spec)
        self.tf_observation_to_action_table = tf.constant(action_function, dtype=tf.float32)
        self.tf_observation_to_update_table = tf.constant(update_function, dtype=tf.float32)
        self.nr_observations = nr_observations if nr_observations is not None else action_function.shape[1]
        self.fix_action_table_probs()
        if descending_actions is not None: # This array contains actions for a given memory and observation in descending order
                                           # The first action is the most likely one, but it could be illegal in some cases
                                           # This array is used to get the most likely legal action, when combined mask
            self.descending_actions = tf.constant(descending_actions, dtype=tf.int32)
        else:
            self.descending_actions = None
        
        self.action_keywords = action_keywords
        self.initial_memory = initial_memory

    def fix_action_table_probs(self):
        """Fixes the action table probabilities to sum to 1 for each observation."""
        if self.nr_observations is not None and self.nr_observations != self.tf_observation_to_action_table.shape[1]:
            # Fill the action table with zeros for the missing observations
            missing_observations = self.nr_observations - self.tf_observation_to_action_table.shape[1]
            if missing_observations > 0:
                zeros = tf.zeros((self.tf_observation_to_action_table.shape[0], missing_observations, self.tf_observation_to_action_table.shape[-1]), dtype=tf.float32)
                self.tf_observation_to_action_table = tf.concat([self.tf_observation_to_action_table, zeros], axis=1)
        if self.tf_observation_to_action_table.shape[-1] != self.tf_observation_to_update_table.shape[-1]:
            # Fill the update table with zeros for the missing updates
            missing_updates = self.tf_observation_to_action_table.shape[-1] - self.tf_observation_to_update_table.shape[-1]
            if missing_updates > 0:
                zeros = tf.zeros((self.tf_observation_to_update_table.shape[0], missing_updates, self.tf_observation_to_update_table.shape[2]), dtype=tf.float32)
                self.tf_observation_to_update_table = tf.concat([self.tf_observation_to_update_table, zeros], axis=1)
        if self.tf_observation_to_action_table.shape[-1] > 1:
            normalizers = tf.reduce_sum(self.tf_observation_to_action_table, axis=-1, keepdims=True)
            self.tf_observation_to_action_table = tf.math.divide_no_nan(self.tf_observation_to_action_table, normalizers)
            # Repair the action table to ensure that it is a valid probability distribution
            self.tf_observation_to_action_table = tf.where(normalizers > 0, 
                                                           self.tf_observation_to_action_table, 
                                                           tf.ones_like(self.tf_observation_to_action_table) / tf.cast(tf.shape(self.tf_observation_to_action_table)[-1], 
                                                                                                                       dtype=tf.float32))
        if self.tf_observation_to_update_table.shape[-1] > 1:
            normalizers = tf.reduce_sum(self.tf_observation_to_update_table, axis=-1, keepdims=True)
            self.tf_observation_to_update_table = tf.math.divide_no_nan(self.tf_observation_to_update_table, normalizers)
            # Repair the update table to ensure that it is a valid probability distribution
            self.tf_observation_to_update_table = tf.where(normalizers > 0, 
                                                           self.tf_observation_to_update_table, 
                                                           tf.ones_like(self.tf_observation_to_update_table) / tf.cast(tf.shape(self.tf_observation_to_update_table)[-1], 
                                                                                                                       dtype=tf.float32))

    def _get_initial_state(self, batch_size):
        return tf.constant(self.initial_memory, shape=(batch_size, 1), dtype=tf.int32)

    @tf.function
    def _action(self, time_step, policy_state, seed):
        observation = time_step.observation["integer"]
        mask = time_step.observation["mask"]
        memory = policy_state
        indices = tf.concat([memory, observation], axis=1)
        if self.descending_actions is not None:
            # Gather the ordered actions for the current observation, remove the illegal actions and select the first legal action
            descending_actions = tf.gather_nd(self.descending_actions, indices)
            descending_actions = tf.boolean_mask(descending_actions, mask)
            action = descending_actions[0]
        else:
            action = tf.gather_nd(self.tf_observation_to_action_table, indices)
        # If actions are arrays of probabilities, we need to sample from them
        if isinstance(action, tf.Tensor) and len(action.shape) > 1:
            action = tf.random.categorical(tf.math.log(action), num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)
        else:
            action = tf.cast(action, dtype=tf.int32)
        update = tf.gather_nd(self.tf_observation_to_update_table, indices)
        
        if isinstance(update, tf.Tensor) and len(update.shape) > 1:
            # Get the indices of the maximum update value for each observation. These indices correspond to the selected update.
            update = tf.random.categorical(tf.math.log(update), num_samples=1, dtype=tf.int32)
            update = tf.reshape(update, (-1, 1))
        else:
            update = tf.reshape(update, (-1, 1))
            update = tf.cast(update, dtype=tf.int32)
        print(f"Action: {action}, Update: {update}, Memory: {memory}")
        policy_step = PolicyStep(action, update)
        return policy_step
    
if __name__ == "__main__":
    args = init_args("models/evade/sketch.templ", "models/evade/sketch.props")

    environment, tf_env = init_environment(args)
    action_keywords = environment.action_keywords
    nr_observations = environment.stormpy_model.nr_observations
    num_fsc_states = 3

    action_function = np.random.uniform(0.1, 1.0, (num_fsc_states, nr_observations, len(action_keywords)))
    update_function = np.random.uniform(0.1, 1.0, (num_fsc_states, nr_observations, num_fsc_states))
    action_function = action_function / np.sum(action_function, axis=-1, keepdims=True) 
    update_function = update_function / np.sum(update_function, axis=-1, keepdims=True)
    
    recurrent_ppo = Recurrent_PPO_agent(environment, tf_env, args)

    policy = TableBasedPolicy(
        original_policy=recurrent_ppo.get_policy(False, True),
        action_function=action_function,
        update_function=update_function,
        initial_memory=0,
        action_keywords=action_keywords,
        descending_actions=None
    )
    eager = PyTFEagerPolicy(policy, use_tf_function=True) # Makes the execution faster, but is not necessary for the policy to work

    evaluate_policy_in_model(policy, args, environment, tf_env, max_steps=400)



