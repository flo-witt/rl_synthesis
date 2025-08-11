# using Paynt for POMDP sketches

import paynt.quotient.fsc
import paynt.synthesizer.synthesizer_ar


import os

from tf_agents.policies import TFPolicy

from environment.environment_wrapper_vec import EnvironmentWrapperVec
from environment.tf_py_environment import TFPyEnvironment

from paynt.rl_extension.self_interpretable_interface.self_interpretable_extractor import SelfInterpretableExtractor

from paynt.rl_extension.family_extractors.direct_fsc_construction import ConstructorFSC

from paynt.rl_extension.robust_rl.family_quotient_numpy import FamilyQuotientNumpy


from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from tools.args_emulator import ArgsEmulator
from tools.evaluators import evaluate_policy_in_model

import numpy as np

import logging

from tools.evaluators import evaluate_policy_in_model

from paynt.quotient.fsc import FscFactored

logger = logging.getLogger(__name__)

from robust_rl.robust_rl_tools import create_json_file_name, assignment_to_pomdp, generate_heatmap_complete
from rl_src.interpreters.bottlenecking.quantized_bottleneck_extractor import BottleneckExtractor

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

        if self.args.extraction_type == "alergia":
            self.autlearn_extraction = True
        else:
            self.autlearn_extraction = False
        self.extraction_type = args.extraction_type
        self.use_gumbel_softmax = use_gumbel_softmax
        self.direct_extractor = self.init_extractor(latent_dim, self.autlearn_extraction)
        
        self.benchmark_stats = self.BenchmarkStats(
            fsc_size=fsc_size, num_training_steps_per_iteration=301, 
            batched_vec_storm=args.batched_vec_storm, extraction_type=args.extraction_type, 
            lstm_width=args.width_of_lstm)
        self.agent = None
        

    class BenchmarkStats:
        def __init__(self, fsc_size=3, num_training_steps_per_iteration=50, batched_vec_storm=True, 
                     extraction_type : str = "alergia", lstm_width=32):
            self.fsc_size = fsc_size
            self.num_training_steps_per_iteration = num_training_steps_per_iteration
            self.rl_performance_single_pomdp = []
            self.rl_performance_single_pomdp_reachability = []
            self.extracted_fsc_return = []
            self.extracted_fsc_reachability = []
            self.family_performance = []
            self.available_nodes_in_fsc = []
            self.worst_case_pomdp_values_simulated_rl = []
            self.worst_case_pomdp_values_simulated_fsc = []
            self.was_worst_case_same_as_simulated = []
            self.worst_case_index_rl = []
            self.worst_case_index_fsc = []
            self.worst_case_index_verif = []
            self.number_of_training_trajectories = []
            self.environment_type = "batched_vec_storm" if batched_vec_storm else "vec_storm"
            self.extraction_type = extraction_type
            self.lstm_width = lstm_width

        def add_rl_performance(self, performance: float):
            self.rl_performance_single_pomdp.append(performance)

        def add_rl_performance_reachability(self, performance: float):
            self.rl_performance_single_pomdp_reachability.append(performance)

        def add_extracted_fsc_performance(self, performance):
            self.extracted_fsc_return.append(performance)
        
        def add_extracted_fsc_reachability(self, performance):
            self.extracted_fsc_reachability.append(performance)

        def add_family_performance(self, performance):
            self.family_performance.append(performance)
        
        def add_worst_case_pomdp_values_simulated_rl(self, value):
            self.worst_case_pomdp_values_simulated_rl.append(value)

        def add_worst_case_pomdp_values_simulated_fsc(self, value):
            self.worst_case_pomdp_values_simulated_fsc.append(value)

        def add_worst_case_same_as_simulated(self, was_same: bool):
            self.was_worst_case_same_as_simulated.append(was_same)

        def add_worst_case_assignments(self, rl_assignment: str, fsc_assignment: str, verif_assignment: str):
            self.worst_case_index_rl.append(rl_assignment)
            self.worst_case_index_fsc.append(fsc_assignment)
            self.worst_case_index_verif.append(verif_assignment)

        def add_number_of_training_trajectories(self, number_of_trajectories: int):
            self.number_of_training_trajectories.append(number_of_trajectories)

    def save_stats(self, path):
        import json
        benchmark_stats = self.benchmark_stats
        stats = {
            "number_of_pomdps_in_family": str(len(list(self.pomdp_sketch.family.all_combinations()))),
            "num_training_steps_per_iteration": str(benchmark_stats.num_training_steps_per_iteration),
            "average_rl_return_subset_simulated": str(benchmark_stats.rl_performance_single_pomdp),
            "average_rl_reachability_subset_simulated": str(benchmark_stats.rl_performance_single_pomdp_reachability),
            "average_extracted_fsc_return_subset_simulated": str(benchmark_stats.extracted_fsc_return),
            "average_extracted_fsc_reachability_subset_simulated": str(benchmark_stats.extracted_fsc_reachability),
            "family_performance": str(benchmark_stats.family_performance),
            "available_nodes_in_fsc": str(benchmark_stats.available_nodes_in_fsc),
            "worst_case_pomdp_values_simulated_rl": str(benchmark_stats.worst_case_pomdp_values_simulated_rl),
            "worst_case_pomdp_values_simulated_fsc": str(benchmark_stats.worst_case_pomdp_values_simulated_fsc),
            "was_worst_case_same_as_simulated": str(benchmark_stats.was_worst_case_same_as_simulated),
            "worst_case_index_rl": str(benchmark_stats.worst_case_index_rl),
            "worst_case_index_fsc": str(benchmark_stats.worst_case_index_fsc),
            "worst_case_index_paynt": str(benchmark_stats.worst_case_index_verif),
            "environment_type": benchmark_stats.environment_type,
            "number_of_extraction_trajectories": str(benchmark_stats.number_of_training_trajectories),
            "extraction_type": benchmark_stats.extraction_type,
            "lstm_width": benchmark_stats.lstm_width
            
        }
        with open(path, 'w') as f:
            json.dump(stats, f, indent=4)

    def init_extractor(self, latent_dim, autlearn_extraction=False):
        if not self.args.extraction_type == "bottleneck":
            direct_extractor = SelfInterpretableExtractor(memory_len=latent_dim, is_one_hot=self.use_one_hot_memory,
                                                        use_residual_connection=True, training_epochs=20001,
                                                        num_data_steps=4001, get_best_policy_flag=False, model_name=self.model_name,
                                                        max_episode_len=self.args.max_steps,
                                                        family_quotient_numpy=self.family_quotient_numpy,
                                                        autlearn_extraction=autlearn_extraction,
                                                        use_gumbel_softmax=self.use_gumbel_softmax)
            return direct_extractor
        else:
            return None
        

    def extract_fsc(self, agent: Recurrent_PPO_agent, environment, quotient, num_data_steps=4001, training_epochs=10001, get_dict = False) -> paynt.quotient.fsc.FscFactored:
        # agent.set_agent_greedy()
        # agent.set_policy_masking()
        agent.set_policy_masking()
        agent.set_agent_stochastic()
        
        policy = agent.get_policy(False, True)
        tf_environment = TFPyEnvironment(environment)
        if self.extraction_type == "bottleneck" or self.direct_extractor is None:
            extractor = BottleneckExtractor(
                tf_environment, 64, 4
            )
            extractor.train_autoencoder(policy, 201, num_data_steps=num_data_steps)
            extractor.evaluate_bottlenecking(
                agent
            )
            fsc = extractor.extract_fsc(
                policy, environment, generate_fake_time_step=self.family_quotient_numpy.get_time_steps_for_observation_integers)
            evaluation_result = evaluate_policy_in_model(
                fsc, self.args, environment, tf_environment)
            self.benchmark_stats.add_extracted_fsc_performance(
                evaluation_result.returns[-1])
            self.benchmark_stats.add_extracted_fsc_reachability(
                evaluation_result.reach_probs[-1])
        else:
            self.direct_extractor.num_data_steps = num_data_steps
            self.direct_extractor.training_epochs = training_epochs
            fsc, extraction_stats = self.direct_extractor.clone_and_generate_fsc_from_policy(
                policy, environment, tf_environment)
            self.benchmark_stats.add_extracted_fsc_performance(
                extraction_stats.extracted_fsc_reward[-1])
            self.benchmark_stats.add_extracted_fsc_reachability(
                extraction_stats.extracted_fsc_reachability[-1])


        print(f"Action shape before construction: {fsc.tf_observation_to_action_table.shape}, updates shape: {fsc.tf_observation_to_update_table.shape}, initial state: {fsc.initial_memory}")
        paynt_fsc = ConstructorFSC.construct_fsc_from_table_based_policy(
            fsc, quotient, family_quotient_numpy=self.family_quotient_numpy)
        available_nodes = paynt_fsc.compute_available_updates(0)
        self.benchmark_stats.available_nodes_in_fsc.append(available_nodes)
        agent.unset_policy_masking()
        agent.set_agent_stochastic()
        
        if get_dict:
            return {
                "extracted_paynt_fsc": paynt_fsc,
                "extracted_fsc": fsc
                }
        return paynt_fsc

    def train_on_new_pomdp(self, pomdp = None, agent: Recurrent_PPO_agent = None, nr_iterations=1500):
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

    def generate_agent(self, pomdp, args : ArgsEmulator) -> Recurrent_PPO_agent:
        self.environment = EnvironmentWrapperVec(pomdp, args, num_envs=args.num_environments, enforce_compilation=True,
                                                 obs_evaluator=self.obs_evaluator,
                                                 quotient_state_valuations=self.quotient_state_valuations,
                                                 observation_to_actions=self.pomdp_sketch.observation_to_actions)
        self.tf_env = TFPyEnvironment(self.environment)
        self.agent = Recurrent_PPO_agent(
            environment=self.environment, tf_environment=self.tf_env, args=args)
        return self.agent
    
    def add_new_pomdp(self, pomdp, agent : Recurrent_PPO_agent):
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

    def sample_data_from_current_environment(self, agent : Recurrent_PPO_agent):
        agent.set_agent_stochastic() # Agent is plays stochastic policy
        agent.set_policy_masking() # Agent uses masking to avoid illegal actions. In the collected data, the illegal action distributions logits are set to some really low constatnt.
        collector_policy = agent.get_policy(False, True) # If the second argument is True, the policy also returns distributions over actions in the info. Useful for stochastic-FSC extraction.
        # Data sampling

    def prepare_environments(self, pomdp_sketch, args_emulated):
        """
        Prepares the environments for the POMDP sketch.
        """
        obs_evaluator = pomdp_sketch.obs_evaluator
        quotient_state_valuations = pomdp_sketch.quotient_mdp.state_valuations
        environment_wrappers = []
        pomdps = []
        for sub_family in pomdp_sketch.family.all_combinations():
            hole_assignment = pomdp_sketch.family.construct_assignment(sub_family)
            pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
            environment = EnvironmentWrapperVec(pomdp, args_emulated, num_envs=self.args.num_environments, enforce_compilation=True,
                                                obs_evaluator=obs_evaluator,
                                                quotient_state_valuations=quotient_state_valuations,
                                                observation_to_actions=pomdp_sketch.observation_to_actions)
            environment_wrappers.append(environment)
            pomdps.append(pomdp)
        tf_environments = [TFPyEnvironment(env) for env in environment_wrappers]
        return environment_wrappers, tf_environments, list(pomdp_sketch.family.all_combinations()), pomdps
        

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
        logger.info(f"Worst hole assignment: {worst_hole_assignment} with value {all_evaluations[worst_hole_assignment]}")
        worst_case_index = all_hole_assignments.index(worst_hole_assignment)
        return worst_case_index
    
    def get_worst_case_pomdp_value(self, all_evaluations : dict):
        worst_hole_value = min(all_evaluations.values())
        return worst_hole_value

    def pure_rl_loop(self, pomdp_sketch, all_evaluations_file="all_evaluations.txt", extract_after_iters=5):
        """
        Pure RL loop, without FSC extraction.
        """

        args_emulated = self.args
        print("All hole assignments in the POMDP sketch:")
        for assignment in pomdp_sketch.family.all_combinations():
            print(assignment)
        environments, tf_environments, all_hole_assignments, pomdps = self.prepare_environments(pomdp_sketch, args_emulated)

        all_evaluations = self.evaluate_on_all_pomdps(self.agent, environments, tf_environments, all_hole_assignments)

        merged_results = self.merge_results(all_evaluations)

        with open(all_evaluations_file, 'w') as f:
            f.write("All evaluations:\n")
            for assignment, values in merged_results.items():
                f.write(f"{assignment}: {values}\n")

        # Get hole assignment, where the agent performed the worst
        
        
        worst_case_index = self.get_worst_case_pomdp_index(all_evaluations, all_hole_assignments)
        for i in range(100):
            logger.info(f"Iteration {i+1} of pure RL loop")
            # Train the agent on all POMDPs
            self.train_on_new_pomdp(pomdps[worst_case_index], self.agent, nr_iterations=501)

            # Evaluate the agent on all POMDPs
            all_evaluations = self.evaluate_on_all_pomdps(self.agent, environments, tf_environments, all_hole_assignments)
            self.benchmark_stats.add_worst_case_pomdp_values_simulated_rl(
                all_evaluations[all_hole_assignments[worst_case_index]])
            merged_results = self.merge_results(all_evaluations, merged_results)
            worst_case_index = self.get_worst_case_pomdp_index(all_evaluations, all_hole_assignments)
            with open(all_evaluations_file, 'w') as f:
                f.write(f"Iteration {i+1} evaluations:\n")
                for assignment, values in merged_results.items():
                    f.write(f"{assignment}: {values}\n")
            if i % extract_after_iters == 0 and i > 0:
                # Extract the FSC from the agent
                fsc = self.extract_fsc(self.agent, self.agent.environment, pomdp_sketch, training_epochs=20001)
                # Evaluate the FSC on all POMDPs
                dtmc_sketch = pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
                synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
                hole_assignment = synthesizer.synthesize(keep_optimum=True)
                logger.info(f"Extracted FSC for hole assignment: {hole_assignment}")
                self.benchmark_stats.add_family_performance(synthesizer.best_assignment_value)
                self.save_stats(os.path.join(f"{args_emulated.model_name}_benchmark_stats.json"))

    def perform_overall_evaluation(self, merged_results, policy, environments, tf_environments, all_hole_assignments, save = False, is_fsc=False, project_path=None):
        all_evaluations = self.evaluate_on_all_pomdps(policy, environments, tf_environments, all_hole_assignments)
        worst_case_value = self.get_worst_case_pomdp_value(all_evaluations)
        if not is_fsc:
            self.benchmark_stats.add_worst_case_pomdp_values_simulated_rl(worst_case_value)
        else:
            self.benchmark_stats.add_worst_case_pomdp_values_simulated_fsc(worst_case_value)
        merged_results = self.merge_results(all_evaluations, merged_results)
        if save:
            suffix = "fsc" if is_fsc else "rl"
            if project_path is not None:
                file_name = os.path.join(project_path, f"{self.args.model_name}_overall_evaluation_{suffix}.txt")
            else:
                file_name = f"{self.args.model_name}_overall_evaluation_{suffix}.txt"
            with open(f"{file_name}", 'w') as f:
                f.write("Overall evaluation:\n")
                for assignment, values in merged_results.items():
                    f.write(f"{assignment}: {values}\n")
        worst_case_index = self.get_worst_case_pomdp_index(all_evaluations, all_hole_assignments)
        return merged_results, worst_case_index
    
    def perform_heatmap_evaluation(self, fsc, pomdp_sketch, save=False, project_path=None):
        """
        Performs a heatmap evaluation of the FSC.
        """
        heatmap_evaluations, hole_assignments_to_test = generate_heatmap_complete(pomdp_sketch, fsc)
        if save:
            if project_path is not None:
                file_name = os.path.join(project_path, f"{self.args.model_name}_heatmap_evaluations.txt")
            else:
                file_name = f"{self.args.model_name}_heatmap_evaluations.txt"
            with open(file_name, 'w') as f:
                f.write(f"Heatmap evaluations for FSC: {fsc}\n")
                for i, evaluation in enumerate(heatmap_evaluations):
                    f.write(f"{hole_assignments_to_test[i]}: {evaluation}\n")
                f.write("\n")
        return heatmap_evaluations, hole_assignments_to_test

    def extraction_loop(self, pomdp_sketch, project_path, nr_initial_pomdps = 10, num_samples_learn=401):
        """
        Pure RL loop, without FSC extraction.
        """
        select_worst_case_by_index = False
        args_emulated = self.args
        
        json_path = create_json_file_name(project_path)
        if select_worst_case_by_index:
            environments, tf_environments, all_hole_assignments, pomdps = self.prepare_environments(pomdp_sketch, args_emulated)
            merged_results, worst_case_index = self.perform_overall_evaluation(None, self.agent.get_policy(False, True), environments, tf_environments, 
                                                                            all_hole_assignments, save=True, project_path=project_path)
            pomdp = pomdps[worst_case_index]
        else:
            hole_assignment = pomdp_sketch.family.pick_random()
            pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)

        if True: # Add 10 random POMDPs to the environment
            for _ in range(nr_initial_pomdps):
                hole_assignment = pomdp_sketch.family.pick_random()
                pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
                self.add_new_pomdp(pomdp, self.agent)
        # from paynt.new_fscs.dpm import get_dpm_fsc
        # from paynt.new_fscs.avoid import get_avoid_fsc
        # fsc = get_avoid_fsc(self.family_quotient_numpy.action_labels.tolist())    
        # fsc = convert_all_fsc_keys_to_int(fsc)
        # dtmc_sketch = pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
        # synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
        # hole_assignment = synthesizer.synthesize(keep_optimum=True)
        # table_based_fsc = generate_table_based_fsc_from_paynt_fsc(fsc, self.environment.action_keywords, self.agent)

        # evaluate_policy_in_model(
        #     table_based_fsc, self.args, self.environment, self.tf_env, self.args.max_steps + 1)
        # exit(0)
        merged_results = None
        nr_iterations = 2001
        for i in range(200):
            logger.info(f"Iteration {i+1} of extraction RL loop")
            # Train the agent on multiple POMDPs
            self.train_on_new_pomdp(pomdp, self.agent, nr_iterations=nr_iterations)
            nr_iterations = 401
            # Evaluate the agent on all POMDPs
            # merged_results, worst_case_index_rl = self.perform_overall_evaluation(merged_results, self.agent.get_policy(False, True), 
            #                                                      environments, tf_environments, all_hole_assignments, save=True,
            #                                                      project_path=project_path)
            fsc = self.extract_fsc(self.agent, self.agent.environment, pomdp_sketch, training_epochs=30001, get_dict=True, num_data_steps=num_samples_learn)
            # Evaluate the FSC on all POMDPs
            paynt_fsc = fsc["extracted_paynt_fsc"]
            # table_based_fsc = fsc["extracted_fsc"]
            # _, worst_case_index_fsc = self.perform_overall_evaluation(merged_results, table_based_fsc, environments, tf_environments, all_hole_assignments, 
            #                                                           save=True, is_fsc=True, project_path=project_path)
            dtmc_sketch = pomdp_sketch.build_dtmc_sketch(paynt_fsc, negate_specification=True)
            synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
            hole_assignment = synthesizer.synthesize(keep_optimum=True)
            

            # self.benchmark_stats.add_worst_case_assignments(all_hole_assignments[worst_case_index_rl], all_hole_assignments[worst_case_index_fsc], hole_assignment)
            # self.benchmark_stats.add_worst_case_same_as_simulated(
            #     None)
            logger.info(f"Extracted FSC for hole assignment: {hole_assignment}")
            self.benchmark_stats.add_family_performance(synthesizer.best_assignment_value)
            self.save_stats(json_path)
            self.agent.evaluation_result.save_to_json(json_path, new_pomdp=True)
            # Perform heatmap evaluation
            # heatmap_evaluations, hole_assignments_to_test = self.perform_heatmap_evaluation(paynt_fsc, pomdp_sketch, save=True, project_path=project_path)

            # Get the worst-case pomdp
            pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
            if i % 2 == 0:
                self.agent.reset_weights() 

def initialize_extractor(pomdp_sketch, args_emulated : ArgsEmulator, family_quotient_numpy : FamilyQuotientNumpy):
    quotient_sv = pomdp_sketch.quotient_mdp.state_valuations
    quotient_obs = pomdp_sketch.obs_evaluator

    quotient_sv = pomdp_sketch.quotient_mdp.state_valuations
    use_one_hot_memory = True if args_emulated.extraction_type == "si-g" else False
    use_gumbel_softmax = True if args_emulated.extraction_type == "si-g" else False
    latent_dim = 9 if use_one_hot_memory else 2 # 3 ** i in case of non-one-hot memory

    # Currently latent_dim is hardcoded to 2 (3 ** i => 9-FSC)
    # If use one-hot memory, then the size of FSC is equal to the latent_dim.
    extractor = RobustTrainer(args_emulated, use_one_hot_memory=use_one_hot_memory, latent_dim=latent_dim, quotient_state_valuations=quotient_sv,
                              obs_evaluator=quotient_obs, pomdp_sketch=pomdp_sketch, 
                              family_quotient_numpy=family_quotient_numpy, use_gumbel_softmax=use_gumbel_softmax)
    
    return extractor
