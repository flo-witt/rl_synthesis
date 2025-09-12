# using Paynt for POMDP sketches

from rl_src.interpreters.bottlenecking.quantized_bottleneck_extractor import BottleneckExtractor
from robust_rl.robust_rl_tools import create_json_file_name, assignment_to_pomdp, generate_heatmap_complete, generate_table_based_fsc_from_paynt_fsc, compute_policy_by_saynt
import paynt.quotient.fsc
import paynt.synthesizer.synthesizer_ar

from robust_rl.rnn_analyzer import RNNAnalyzer

import os

from tf_agents.policies import TFPolicy

from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from rl_src.environment.tf_py_environment import TFPyEnvironment

from paynt.rl_extension.self_interpretable_interface.self_interpretable_extractor import SelfInterpretableExtractor

from paynt.rl_extension.family_extractors.direct_fsc_construction import ConstructorFSC

from paynt.rl_extension.robust_rl.family_quotient_numpy import FamilyQuotientNumpy
from paynt.quotient.pomdp import PomdpQuotient
from paynt.quotient.pomdp_family import PomdpFamilyQuotient


from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.tools.evaluators import evaluate_policy_in_model

import numpy as np

import logging


from robust_rl.benchmark_stats import BenchmarkStats

from paynt.quotient.fsc import FscFactored

from robust_rl.config_file import Config


logger = logging.getLogger(__name__)


class RobustTrainer:
    def __init__(self, args: ArgsEmulator, use_one_hot_memory=False, latent_dim=2,
                 pomdp_sketch=None,
                 obs_evaluator=None, quotient_state_valuations=None,
                 family_quotient_numpy: FamilyQuotientNumpy = None,
                 use_gumbel_softmax=False):
        self.args = args
        self.use_one_hot_memory = use_one_hot_memory
        self.model_name = args.model_name
        self.pomdp_sketch = pomdp_sketch
        self.obs_evaluator = obs_evaluator
        self.quotient_state_valuations = quotient_state_valuations
        self.family_quotient_numpy = family_quotient_numpy
        fsc_size = latent_dim if use_one_hot_memory else 3**latent_dim
        self.cut_probs = 0.0

        if self.args.extraction_type == "alergia":
            self.autlearn_extraction = True
        else:
            self.autlearn_extraction = False
        self.extraction_type = args.extraction_type
        self.use_gumbel_softmax = use_gumbel_softmax
        self.direct_extractor = self.init_extractor(
            latent_dim, self.autlearn_extraction)
        self.period_between_worst_case_evaluation = 5

        self.benchmark_stats = BenchmarkStats(
            fsc_size=fsc_size, num_training_steps_per_iteration=301,
            batched_vec_storm=args.batched_vec_storm, extraction_type=args.extraction_type,
            lstm_width=args.width_of_lstm, without_extraction=args.without_extraction,
            geometric_batched_vec_storm=args.geometric_batched_vec_storm,
            periodic_restarts=args.periodic_restarts, period_between_worst_case_evaluation=self.period_between_worst_case_evaluation,
            seed=args.seed)
        self.agent = None
        self.extraction_less = args.without_extraction
        self.fscs_extracted = []

    def save_stats(self, path):
        self.benchmark_stats.save_stats(path)
    



    def init_extractor(self, latent_dim, autlearn_extraction=False):
        if not self.args.extraction_type == "bottleneck":
            direct_extractor = SelfInterpretableExtractor(memory_len=latent_dim, is_one_hot=self.use_one_hot_memory,
                                                          use_residual_connection=True, training_epochs=20001,
                                                          num_data_steps=self.args.max_steps * 4, get_best_policy_flag=False, model_name=self.model_name,
                                                          max_episode_len=self.args.max_steps,
                                                          family_quotient_numpy=self.family_quotient_numpy,
                                                          autlearn_extraction=autlearn_extraction,
                                                          use_gumbel_softmax=self.use_gumbel_softmax)
            return direct_extractor
        else:
            return None

    def call_bottlenecking(self, agent: Recurrent_PPO_agent, environment: EnvironmentWrapperVec, tf_environment: TFPyEnvironment, num_data_steps=4001, training_epochs=10001):
        policy = agent.get_policy(False, True)
        extractor = BottleneckExtractor(
            tf_environment, 64, 4
        )
        extractor.train_autoencoder(
            policy, 61, num_data_steps=num_data_steps)
        extractor.evaluate_bottlenecking(
            agent
        )
        fsc = extractor.extract_fsc(
            policy, environment,
            generate_fake_time_step=self.family_quotient_numpy.get_time_steps_for_observation_integers,
            nr_observations=len(self.family_quotient_numpy.observation_to_legal_action_mask))
        evaluation_result = evaluate_policy_in_model(
            fsc, self.args, environment, tf_environment)
        self.benchmark_stats.add_extracted_fsc_performance(
            evaluation_result.returns[-1])
        self.benchmark_stats.add_extracted_fsc_reachability(
            evaluation_result.reach_probs[-1])
        return fsc
    
    def call_si_or_aalpy(self, agent: Recurrent_PPO_agent, environment: EnvironmentWrapperVec, tf_environment: TFPyEnvironment, num_data_steps=4001, training_epochs=10001):
        policy = agent.get_policy(False, True)
        fsc, extraction_stats = self.direct_extractor.clone_and_generate_fsc_from_policy(
            policy, environment, tf_environment)
        self.benchmark_stats.add_extracted_fsc_performance(
            extraction_stats.extracted_fsc_reward[-1])
        self.benchmark_stats.add_extracted_fsc_reachability(
            extraction_stats.extracted_fsc_reachability[-1])
        if len(extraction_stats.lstm_extracted_reachability) > 0 and len(extraction_stats.lstm_extracted_return) > 0:
            self.benchmark_stats.add_lstm_extracted_results(
                extraction_stats.lstm_extracted_reachability[-1], extraction_stats.lstm_extracted_return[-1])
        return fsc

    def extract_fsc(self, agent: Recurrent_PPO_agent, environment: EnvironmentWrapperVec, quotient, num_data_steps=4001, training_epochs=10001, get_dict=False,
                    use_masking : bool = True) -> paynt.quotient.fsc.FscFactored:
        self.direct_extractor.num_data_steps = num_data_steps
        self.direct_extractor.training_epochs = training_epochs
        # agent.set_agent_greedy()
        # agent.set_policy_masking()
        if use_masking:
            agent.set_policy_masking()
        else:
            agent.unset_policy_masking()


        tf_environment = TFPyEnvironment(environment)
        # if True:
        #     from rl_src.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy, initialize_random_joint_fsc_function
        #
        #     action_function, update_function = initialize_random_joint_fsc_function(
        #         environment.action_keywords, len(self.family_quotient_numpy.observation_to_legal_action_mask), 3)
        #     fsc = TableBasedPolicy(
        #         original_policy=policy, action_function=action_function, update_function=update_function,
        #         action_keywords=environment.action_keywords, nr_observations=len(self.family_quotient_numpy.observation_to_legal_action_mask))

        if self.extraction_type == "bottleneck" or self.direct_extractor is None:
            fsc = self.call_bottlenecking(
                agent, environment, tf_environment, num_data_steps=num_data_steps, training_epochs=training_epochs)
        else:
            fsc = self.call_si_or_aalpy(
                agent, environment, tf_environment, num_data_steps=num_data_steps, training_epochs=training_epochs)
        # from paynt.rl_extension.self_interpretable_interface.tabular_reinforce import TabularReinforce
        # TabularReinforce.train(fsc, environment)
        paynt_fsc = ConstructorFSC.construct_fsc_from_table_based_policy(
            fsc, quotient, family_quotient_numpy=self.family_quotient_numpy, cut_probs=self.cut_probs)
        self.fscs_extracted.append(paynt_fsc)
        # policy_to_test = generate_table_based_fsc_from_paynt_fsc(
        #     paynt_fsc, environment.action_keywords, agent)
        # evaluation_result = evaluate_policy_in_model(
        #     policy_to_test, self.args, environment, tf_environment)

        available_nodes = paynt_fsc.compute_available_updates(0)
        self.benchmark_stats.available_nodes_in_fsc.append(available_nodes)

        if get_dict:
            return {
                "extracted_paynt_fsc": paynt_fsc,
                "extracted_fsc": fsc
            }
        return paynt_fsc

    def train_on_new_pomdp(self, pomdp=None, agent: Recurrent_PPO_agent = None, nr_iterations=1500):
        # environment = EnvironmentWrapperVec(pomdp, self.args, num_envs=256, enforce_compilation=True,
        #                                     obs_evaluator=self.obs_evaluator,
        #                                     quotient_state_valuations=self.quotient_state_valuations,
        #                                     observation_to_actions=self.pomdp_sketch.observation_to_actions)
        if pomdp is not None and self.args.batched_vec_storm:
            self.environment.add_new_pomdp(pomdp)
        elif pomdp is not None and not self.args.batched_vec_storm:
            self.environment = EnvironmentWrapperVec(pomdp, self.args, num_envs=self.args.num_environments, enforce_compilation=True,
                                                     obs_evaluator=self.obs_evaluator,
                                                     quotient_state_valuations=self.quotient_state_valuations,
                                                     observation_to_actions=self.pomdp_sketch.observation_to_actions)
            agent.change_environment(self.environment)
        else:
            logger.info("No POMDP provided, using existing environment.")
        agent.train_agent(iterations=nr_iterations)
        self.benchmark_stats.add_rl_performance(
            np.abs(agent.evaluation_result.returns[-1]))
        self.benchmark_stats.add_rl_performance_reachability(
            np.abs(agent.evaluation_result.reach_probs[-1]))

    def generate_agent(self, pomdp, args: ArgsEmulator) -> Recurrent_PPO_agent:
        self.environment = EnvironmentWrapperVec(pomdp, args, num_envs=args.num_environments, enforce_compilation=True,
                                                 obs_evaluator=self.obs_evaluator,
                                                 quotient_state_valuations=self.quotient_state_valuations,
                                                 observation_to_actions=self.pomdp_sketch.observation_to_actions)
        self.tf_env = TFPyEnvironment(self.environment)
        self.agent = Recurrent_PPO_agent(
            environment=self.environment, tf_environment=self.tf_env, args=args)
        return self.agent

    def add_new_pomdp(self, pomdp, agent: Recurrent_PPO_agent):
        """
        Adds a new POMDP to the environment.
        """
        if self.args.batched_vec_storm:
            agent.environment.add_new_pomdp(pomdp)
        else:
            agent.environment = EnvironmentWrapperVec(pomdp, self.args, num_envs=256, enforce_compilation=True,
                                                      obs_evaluator=self.obs_evaluator,
                                                      quotient_state_valuations=self.quotient_state_valuations,
                                                      observation_to_actions=self.pomdp_sketch.observation_to_actions)
            agent.change_environment(self.environment)

    def prepare_environments(self, pomdp_sketch, args_emulated):
        """
        Prepares the environments for the POMDP sketch.
        """
        obs_evaluator = pomdp_sketch.obs_evaluator
        quotient_state_valuations = pomdp_sketch.quotient_mdp.state_valuations
        environment_wrappers = []
        pomdps = []
        for sub_family in pomdp_sketch.family.all_combinations():
            hole_assignment = pomdp_sketch.family.construct_assignment(
                sub_family)
            pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
            environment = EnvironmentWrapperVec(pomdp, args_emulated, num_envs=self.args.num_environments, enforce_compilation=True,
                                                obs_evaluator=obs_evaluator,
                                                quotient_state_valuations=quotient_state_valuations,
                                                observation_to_actions=pomdp_sketch.observation_to_actions)
            environment_wrappers.append(environment)
            pomdps.append(pomdp)
        tf_environments = [TFPyEnvironment(env)
                           for env in environment_wrappers]
        return environment_wrappers, tf_environments, list(pomdp_sketch.family.all_combinations()), pomdps

    def prepare_subset_environments_for_evaluation(self, pomdp_sketch, pomdps, args_emulated) -> tuple[list[EnvironmentWrapperVec], list[TFPyEnvironment]]:
        """
        Prepares the subset of POMDPs for worst-case evaluation.
        """
        obs_evaluator = pomdp_sketch.obs_evaluator
        quotient_state_valuations = pomdp_sketch.quotient_mdp.state_valuations
        environment_wrappers = []
        for pomdp in pomdps:
            environment = EnvironmentWrapperVec(pomdp, args_emulated, num_envs=self.args.num_environments, enforce_compilation=True,
                                                obs_evaluator=obs_evaluator,
                                                quotient_state_valuations=quotient_state_valuations,
                                                observation_to_actions=pomdp_sketch.observation_to_actions)
            environment_wrappers.append(environment)
        tf_environments = [TFPyEnvironment(env)
                           for env in environment_wrappers]
        return environment_wrappers, tf_environments

    def add_pomdp_to_subset(self, pomdp, environments: list[EnvironmentWrapperVec], tf_environments: list[TFPyEnvironment]):
        """
        Adds a new POMDP to the subset of environments.
        """
        environment = EnvironmentWrapperVec(pomdp, self.args, num_envs=self.args.num_environments, enforce_compilation=True,
                                            obs_evaluator=self.obs_evaluator,
                                            quotient_state_valuations=self.quotient_state_valuations,
                                            observation_to_actions=self.pomdp_sketch.observation_to_actions)
        environments.append(environment)
        tf_environments.append(TFPyEnvironment(environment))

    def evaluate_on_all_pomdps(self, policy: TFPolicy, environments, tf_environments, hole_assignments):
        results = {}
        for i, environment in enumerate(environments):
            result = evaluate_policy_in_model(
                policy, self.args, environment, tf_environments[i], self.args.max_steps + 1)
            results[hole_assignments[i]] = result.best_return
        return results

    def merge_results(self, results, merged_results=None):
        """
        Merges the results from the evaluation of all POMDPs.
        """
        if merged_results is None:
            merged_results = {}
        for assignment, value in results.items():
            if assignment not in merged_results:
                merged_results[assignment] = []
            merged_results[assignment].append(value)
        return merged_results

    def get_worst_case_pomdp_index(self, all_evaluations, all_hole_assignments):
        worst_hole_assignment = min(all_evaluations, key=all_evaluations.get)
        logger.info(
            f"Worst hole assignment: {worst_hole_assignment} with value {all_evaluations[worst_hole_assignment]}")
        worst_case_index = all_hole_assignments.index(worst_hole_assignment)
        return worst_case_index

    def get_worst_case_pomdp_value(self, all_evaluations: dict):
        worst_hole_value = min(all_evaluations.values())
        return worst_hole_value

    def perform_subset_evaluation(self, policy, environments, tf_environments):
        """
        Performs evaluation on a subset of POMDPs.
        """
        results_returns = []
        results_reachabilities = []
        for i, environment in enumerate(environments):
            result = evaluate_policy_in_model(
                policy, self.args, environment, tf_environments[i], self.args.max_steps + 1)
            results_returns.append(result.best_return)
            results_reachabilities.append(result.best_reach_prob)
        return results_returns, results_reachabilities

    def perform_heatmap_evaluation(self, fsc, pomdp_sketch, save=False, project_path=None):
        """
        Performs a heatmap evaluation of the FSC.
        """
        heatmap_evaluations, hole_assignments_to_test = generate_heatmap_complete(
            pomdp_sketch, fsc)
        if save:
            if project_path is not None:
                file_name = os.path.join(
                    project_path, f"{self.args.model_name}_heatmap_evaluations.txt")
            else:
                file_name = f"{self.args.model_name}_heatmap_evaluations.txt"
            with open(file_name, 'w') as f:
                f.write(f"Heatmap evaluations for FSC: {fsc}\n")
                for i, evaluation in enumerate(heatmap_evaluations):
                    f.write(f"{hole_assignments_to_test[i]}: {evaluation}\n")
                f.write("\n")
        return heatmap_evaluations, hole_assignments_to_test

    def extraction_loop(self, pomdp_sketch, project_path, nr_initial_pomdps=10, num_samples_learn=401):
        """
        Pure RL loop, without FSC extraction.
        """
        operator = min if pomdp_sketch.specification.optimality.minimizing else max
        logger.info("Starting extraction loop")
        rnn_analyzer = RNNAnalyzer(self.args)
        select_worst_case_by_index = False
        args_emulated = self.args
        training_epochs = 6001 if "avoid-large" in project_path else 5001
        if project_path.split("/")[-1] == "":
            config = Config(project_path.split("/")[-2])
        else:
            config = Config(project_path.split("/")[-1])
        json_path = create_json_file_name(f"{project_path}", seed=f"{self.args.seed}")

        hole_assignment = pomdp_sketch.family.pick_random()

        pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
        pomdps = [pomdp]
        if True:  # Add nr_initial_pomdps random POMDPs to the environment
            # nr_initial_pomdps = nr_initial_pomdps if not self.extraction_less else 50
            for _ in range(nr_initial_pomdps):
                hole_assignment = pomdp_sketch.family.pick_random()
                pomdp, _, _ = assignment_to_pomdp(
                    pomdp_sketch, hole_assignment)
                self.add_new_pomdp(pomdp, self.agent)
                pomdps.append(pomdp)
        environments, tf_environments = self.prepare_subset_environments_for_evaluation(
            pomdp_sketch, pomdps, args_emulated)
        nr_iterations = config.nr_initial_iter
        for i in range(200):
            logger.info(f"Iteration {i+1} of extraction RL loop")
            # Train the agent on multiple POMDPs
            if args_emulated.single_pomdp_experiment:
                pomdp = None
            self.train_on_new_pomdp(
                pomdp, self.agent, nr_iterations=nr_iterations)
            nr_clusters = rnn_analyzer.analyze(self.agent, self.tf_env)
            self.benchmark_stats.add_nr_clusters(nr_clusters)
            nr_iterations = config.nr_inner_iter if not self.args.periodic_restarts else 401
            # Evaluate the agent on all POMDPs
            # merged_results, worst_case_index_rl = self.perform_overall_evaluation(merged_results, self.agent.get_policy(False, True),
            #                                                      environments, tf_environments, all_hole_assignments, save=True,
            #                                                      project_path=project_path)
            best_assignment_value = None
            for use_masking in [True]:

                fsc = self.extract_fsc(self.agent, self.agent.environment, pomdp_sketch, get_dict=True, num_data_steps=num_samples_learn, use_masking=use_masking, training_epochs=501)
                # Evaluate the FSC on all POMDPs
                paynt_fsc = fsc["extracted_paynt_fsc"]
                table_based_fsc = fsc["extracted_fsc"]
                
                # _, worst_case_index_fsc = self.perform_overall_evaluation(merged_results, table_based_fsc, environments, tf_environments, all_hole_assignments,
                #                                                           save=True, is_fsc=True, project_path=project_path)
                dtmc_sketch = pomdp_sketch.build_dtmc_sketch(
                    paynt_fsc, negate_specification=True)
                synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(
                    dtmc_sketch)
                hole_assignment = synthesizer.synthesize(keep_optimum=True)
                
                best_assignment_value = synthesizer.best_assignment_value if best_assignment_value is None else operator(best_assignment_value, synthesizer.best_assignment_value)

            # self.benchmark_stats.add_worst_case_assignments(all_hole_assignments[worst_case_index_rl], all_hole_assignments[worst_case_index_fsc], hole_assignment)
            # self.benchmark_stats.add_worst_case_same_as_simulated(
            #     None)
            logger.info(
                f"Extracted FSC for hole assignment: {hole_assignment}")
            self.benchmark_stats.add_family_performance(
                synthesizer.best_assignment_value)

            # Perform heatmap evaluation
            # heatmap_evaluations, hole_assignments_to_test = self.perform_heatmap_evaluation(paynt_fsc, pomdp_sketch, save=True, project_path=project_path)
            if not self.extraction_less:
                pomdp, _, _ = assignment_to_pomdp(
                    pomdp_sketch, hole_assignment)
            else:
                hole_assignment = pomdp_sketch.family.pick_random()
                pomdp, _, _ = assignment_to_pomdp(
                    pomdp_sketch, hole_assignment)
            self.add_pomdp_to_subset(pomdp, environments, tf_environments)
            pomdps.append(pomdp)

            last_extracted_fsc_return = np.abs(
                self.benchmark_stats.extracted_fsc_return[-1])
            last_rl_return = np.abs(
                self.benchmark_stats.rl_performance_single_pomdp[-1])
            if self.args.shrink_and_perturb_externally and np.abs(last_extracted_fsc_return - last_rl_return) / ((np.abs(last_extracted_fsc_return) + np.abs(last_rl_return)) / 2.0) > 0.3:
                    self.agent.shrink_and_perturb()
                    nr_iterations = 101
                    self.benchmark_stats.shrink_and_perturb_activated.append(
                        True)
            else:
                    self.benchmark_stats.shrink_and_perturb_activated.append(
                        False)

            self.agent.evaluation_result.save_to_json(
                json_path, new_pomdp=True)

            self.save_stats(json_path)
            if self.args.periodic_restarts:
                self.agent.reset_weights()

    def train_and_extract_single_pomdp(self, pomdp_sketch : PomdpFamilyQuotient, nr_iterations=1500, num_samples_learn=4001, args: ArgsEmulator = None, project_path: str = None):
        """
        RL training and extraction on a single POMDP. Loopless.
        """
        rnn_analyzer = RNNAnalyzer(self.args)
        
        self.train_on_new_pomdp(None, self.agent, nr_iterations=nr_iterations)
        rnn_analyzer.analyze(self.agent, self.tf_env)
        fsc = self.extract_fsc(self.agent, self.agent.environment, pomdp_sketch, num_data_steps=num_samples_learn, training_epochs=501, get_dict=True)
        paynt_fsc = fsc["extracted_paynt_fsc"]
        import pickle as pkl
        with open(os.path.join(project_path, "extracted_fsc.pkl"), "wb") as f:
            pkl.dump(paynt_fsc, f)
            
        dtmc_sketch = pomdp_sketch.build_dtmc_sketch(
            paynt_fsc)
        one_by_one = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(
            dtmc_sketch)
        hole_assignment = one_by_one.synthesize(keep_optimum=True)
        self.benchmark_stats.add_family_performance(
                one_by_one.best_assignment_value)
        logger.info(f"Synthesized assignment: {hole_assignment} with value {one_by_one.best_assignment_value}")
        json_path = create_json_file_name(f"{project_path}", seed=f"{self.args.seed}")
        self.agent.evaluation_result.save_to_json(json_path, new_pomdp=False)
        self.save_stats(json_path)
        return paynt_fsc, hole_assignment, one_by_one.best_assignment_value
    




def initialize_extractor(pomdp_sketch, args_emulated: ArgsEmulator, family_quotient_numpy: FamilyQuotientNumpy):
    if family_quotient_numpy is not None:
        quotient_sv = pomdp_sketch.quotient_mdp.state_valuations
        quotient_obs = pomdp_sketch.obs_evaluator
    else:
        quotient_sv = None
        quotient_obs = None

    use_one_hot_memory = True if args_emulated.extraction_type == "si-g" else False
    use_gumbel_softmax = True if args_emulated.extraction_type == "si-g" else False
    # 3 ** i in case of non-one-hot memory
    latent_dim = 3

    # Currently latent_dim is hardcoded to 2 (3 ** i => 9-FSC)
    # If use one-hot memory, then the size of FSC is equal to the latent_dim.
    extractor = RobustTrainer(args_emulated, use_one_hot_memory=use_one_hot_memory, latent_dim=latent_dim, quotient_state_valuations=quotient_sv,
                              obs_evaluator=quotient_obs, pomdp_sketch=pomdp_sketch,
                              family_quotient_numpy=family_quotient_numpy, use_gumbel_softmax=use_gumbel_softmax)

    return extractor
