# Implementation of PPO agent with recurrent neural networks.
# Author: David Hudák
# Login: xhudak03
# File: recurrent_ppo_agent.py

import logging
from agents.father_agent import FatherAgent
from tools.encoding_methods import *

import tensorflow as tf

from environment import tf_py_environment
# from tf_agents.agents.ppo import ppo_agent
from agents.tf_agents_modif import ppo_agent



from environment.environment_wrapper import Environment_Wrapper

from agents.policies.policy_mask_wrapper import PolicyMaskWrapper

from agents.networks.value_networks import create_recurrent_value_net_demasked
from agents.networks.actor_networks import create_recurrent_actor_net_demasked
from agents.networks.fsc_like_network import FSCLikeNetwork

from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from agents.tf_agents_modif.actor_distribution_rnn_network import ActorDistributionRnnNetwork

from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

from agents.alternative_training.active_pretraining import EntropyRewardGenerator

from tools.args_emulator import ArgsEmulator

from keras.optimizers import Adam


import sys
sys.path.append("../")


logger = logging.getLogger(__name__)



class Recurrent_PPO_agent(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 args : ArgsEmulator, load=False, agent_folder=None, actor_net: ActorDistributionRnnNetwork = None,
                 critic_net: ValueRnnNetwork = None, discrete_actor_memory: bool = False):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        train_step_counter = tf.Variable(0)
        optimizer = Adam(
            learning_rate=args.learning_rate, beta_1=0.99, beta_2=0.99, weight_decay=0.001)
        tf_environment = self.tf_environment
        action_spec = tf_environment.action_spec()
        if actor_net is not None:
            self.actor_net = actor_net
        elif discrete_actor_memory:
            self.actor_net = FSCLikeNetwork(
                tf_environment.observation_spec()["observation"],
                action_spec,
                memory_length=2,
                lstm_size=(32,),
                input_fc_layer_params=(64,),
            )
        else:
            self.actor_net = create_recurrent_actor_net_demasked(
                tf_environment, action_spec, rnn_less=self.args.use_rnn_less, width_of_lstm=self.args.width_of_lstm)
        if critic_net is not None:
            self.value_net = critic_net
        else:
            self.value_net = create_recurrent_value_net_demasked(
                tf_environment, rnn_less=self.args.use_rnn_less, width_of_lstm=self.args.width_of_lstm)

        time_step_spec = tf_environment.time_step_spec()
        time_step_spec = time_step_spec._replace(
            observation=tf_environment.observation_spec()["observation"])
        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            optimizer,
            actor_net=self.actor_net,
            value_net=self.value_net,
            num_epochs=3,
            train_step_counter=train_step_counter,
            greedy_eval=self.args.completely_greedy,
            discount_factor=self.args.discount_factor,
            use_gae=True,
            lambda_value=0.95,
            # policy_l2_reg=0.001,
            # value_function_l2_reg=0.001,
            entropy_regularization=0.02,
            normalize_rewards=True,
            normalize_observations=True,
            importance_ratio_clipping=0.2,
        )
        self.agent.initialize()
        
        logging.info("Agent initialized")
        self.init_replay_buffer()
        logging.info("Replay buffer initialized")

        self.init_collector_driver(self.tf_environment, demasked=True)
        if self.args.predicate_automata_obs or self.args.curiosity_automata_reward or self.args.go_explore:
            predicate_automata = self.environment.predicate_automata
        else:
            predicate_automata = None
            
        self.wrapper = PolicyMaskWrapper(self.agent.policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec(),
                                           is_greedy=(not self.args.prefer_stochastic), predicate_automata=predicate_automata)
        self.wrapper.set_policy_masker()
        self.wrapper_eager = PyTFEagerPolicy(self.wrapper, True, False)
        logging.info("Collector driver initialized")
        if self.args.entropy_reward:
            entropy_reward_generator = EntropyRewardGenerator(binary_flag=args.use_binary_entropy_reward, 
                                                              full_observability_flag=args.full_observable_entropy_reward, 
                                                              max_reward=1.0, decreaser='halve')
            self.init_pretraining_driver(entropy_reward_generator)
        if load:
            self.load_agent()
        # self.init_vec_evaluation_driver(
        #     self.tf_environment, self.environment, num_steps=self.args.max_steps)
        # logging.info("Evaluation driver initialized")        

        
    def special_agent_pretraining_stuff(self):
        logger.info("Setting value net to trainable")
        for var in self.agent.trainable_variables:
            var._trainable = False
        for var in self.agent._value_net.variables:
            var._trainable = True

    def special_agent_midtraining_stuff(self):
        logger.info("Setting actor net to trainable")
        for var in self.agent.variables:
            var._trainable = True
        self.agent.initialize()

    def set_policy_masking(self):
        """If PPO, this function sets the masking active for agent wrapper."""
        self.wrapper.set_policy_masker()

    def unset_policy_masking(self):
        """If PPO, this function sets the masking inactive for agent wrapper."""
        self.wrapper.unset_policy_masker()

    def reset_weights(self):
        for var in self.agent.variables:
            if "kernel" in var.name:
                if "dynamic_unroll" in var.name:  # Rekurentní vrstvy
                    # Použití Glorot Uniform pro RNN váhy
                    glorot_stddev = tf.sqrt(2.0 / (var.shape[0] + var.shape[1]))
                    var.assign(tf.random.uniform(var.shape, minval=-glorot_stddev, maxval=glorot_stddev))
                else:  # Dense vrstvy
                    # Použití Variance Scaling s truncated normal pro Dense váhy
                    scale = 2.0
                    stddev = tf.sqrt(scale / var.shape[0])
                    var.assign(tf.random.truncated_normal(var.shape, stddev=stddev))
            elif "bias" in var.name:
                # Inicializace biasů na nulu
                var.assign(tf.zeros(var.shape))
        self.agent.initialize()



