
import paynt.cli
import paynt.synthesizer.synthesizer_ar

import os
import cProfile
import pstats

import paynt.synthesizer.synthesizer_onebyone
import paynt.utils
import paynt.utils.timer

from tests.general_test_tools import init_args

from paynt.rl_extension.robust_rl.family_quotient_numpy import FamilyQuotientNumpy

import logging

logger = logging.getLogger(__name__)

from robust_rl.robust_rl_tools import generate_heatmap_complete, assignment_to_pomdp, create_json_file_name, load_sketch
from robust_rl.robust_rl_tools import parse_args

from robust_rl.robust_rl_trainer import RobustTrainer, initialize_extractor





def main():
    args_cmd = parse_args()

    paynt.utils.timer.GlobalTimer.start()

    profiling = True
    if profiling:
        pr = cProfile.Profile()
        pr.enable()

    paynt.cli.setup_logger()

    project_path = args_cmd.project_path
    pomdp_sketch = load_sketch(project_path);
    json_path = create_json_file_name(project_path)


    num_samples_learn = 802
    nr_pomdps = 10
    autlearn_extraction = False

    family_quotient_numpy = FamilyQuotientNumpy(pomdp_sketch) # This can be useful for extraction and some other stuff.
    
    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")

    # Here, you can change the main parameters of the training etc.
    # Batched_vec_storm is used to run multiple different POMDPs in parallel. If you want to always run a single POMDP (e.g. worst-case), set it to False.
    # Masked_training is used to train the agent with masking, i.e. the agent will be forbidden to take illegal actions.
    args_emulated = init_args(
        prism_path=prism_path, properties_path=properties_path, batched_vec_storm=True, masked_training=False)
    args_emulated.width_of_lstm = args_cmd.lstm_width
    args_emulated.batched_vec_storm = args_cmd.batched_vec_storm
    args_emulated.extraction_type = args_cmd.extraction_method
    args_emulated.model_name = project_path.split("/")[-1]
    args_emulated.max_steps = 801
    # pomdp = initialize_prism_model(prism_path, properties_path, constants="")

    hole_assignment = pomdp_sketch.family.pick_any()
    pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)

    extractor = initialize_extractor(pomdp_sketch, args_emulated, family_quotient_numpy)

    agent = extractor.generate_agent(pomdp, args_emulated)
    last_hole = None
    for i in range(nr_pomdps):
        hole_assignment = pomdp_sketch.family.pick_random()
    extractor.extraction_loop(pomdp_sketch, project_path=project_path, nr_initial_pomdps=nr_pomdps, num_samples_learn=num_samples_learn)
    return
    # if train_extraction_less_and_then_extract:
    #     file_name_suffix = "batched" if args_emulated.batched_vec_storm else "single"
    #     file_name_suffix += "_masked" if args_emulated.masked_training else "_unmasked"
    #     file_name_suffix += "_rnn_less" if args_emulated.use_rnn_less else "_rnn"
    #     file_name = os.path.join(project_path, f"extraction_evaluations_{file_name_suffix}.txt")
    #     extractor.pure_rl_loop(pomdp_sketch, all_evaluations_file=file_name, extract_after_iters = 5)
    #     return
    # else:
    #     file_name_suffix = "batched" if args_emulated.batched_vec_storm else "single"
    #     file_name_suffix += "_masked" if args_emulated.masked_training else "_unmasked"
    #     file_name_suffix += "_rnn_less" if args_emulated.use_rnn_less else "_rnn"
    #     file_name = os.path.join(project_path, f"all_evaluations_{file_name_suffix}.txt")
    #     extractor.pure_rl_loop(pomdp_sketch, all_evaluations_file=file_name)
    #     return
    last_hole = None
    for i in range(nr_pomdps):
        hole_assignment = pomdp_sketch.family.pick_random()
        pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
        extractor.add_new_pomdp(pomdp,agent)
        last_hole = hole_assignment

    

    # Using None for the POMDP here means that the extractor will use the existing environment from initialization.
    extractor.train_on_new_pomdp(None, agent, nr_iterations=501)

    # -------------------------------------------------------------------------
    # Different extraction method should be used here!!!
    fsc = extractor.extract_fsc(
        agent, agent.environment, training_epochs=20001, quotient=pomdp_sketch, num_data_steps=num_samples_learn)
    # -------------------------------------------------------------------------
    dtmc_sketch = pomdp_sketch.build_dtmc_sketch(
        fsc, negate_specification=True)
    
    heatmap_evaluations, hole_assignments_to_test = generate_heatmap_complete(pomdp_sketch, fsc)
    with open(os.path.join(project_path, "heatmap_evaluations.txt"), 'w') as f:
        f.write(f"Heatmap evaluations, the original assignment is: {hole_assignment}\n", )
        for i, evaluation in enumerate(heatmap_evaluations):
            f.write(f"{hole_assignments_to_test[i]}: {evaluation}\n")
        f.write("\n")



    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
    hole_assignment = synthesizer.synthesize(keep_optimum=True)
    print("Best value", synthesizer.best_assignment_value)
    

    extractor.benchmark_stats.add_family_performance(
        synthesizer.best_assignment_value)
    extractor.save_stats(json_path)
    

    for i in range(50):
        pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)

        extractor.train_on_new_pomdp(pomdp, agent, nr_iterations=501)
        # -------------------------------------------------------------------------
        # Different extraction method should be used here!!!
        fsc = extractor.extract_fsc(
            agent, agent.environment, quotient=pomdp_sketch, num_data_steps=3001, training_epochs=10001)
        # -------------------------------------------------------------------------

        dtmc_sketch = pomdp_sketch.build_dtmc_sketch(
            fsc, negate_specification=True)
        synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(
            dtmc_sketch)
        with open(os.path.join(project_path, "heatmap_evaluations.txt"), 'a') as f:
            f.write(f"Heatmap evaluations, the new assignment is: {hole_assignment}\n")
            for i, evaluation in enumerate(heatmap_evaluations):
                f.write(f"{hole_assignments_to_test[i]}: {evaluation}\n")
            f.write("\n")
        hole_assignment = synthesizer.synthesize(keep_optimum=True)

        one_by_one = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(dtmc_sketch)
        worst_case_eval = one_by_one.evaluate(last_hole, keep_value_only=True)
        print()
        print(f"HOLES:{hole_assignment}")
        print("Best value: ", synthesizer.best_assignment_value)
        print("Worst-case eval last hole:", worst_case_eval)
        if hole_assignment is None:
            continue
        print("Best value", synthesizer.best_assignment_value)
        extractor.benchmark_stats.add_family_performance(
            synthesizer.best_assignment_value)
        extractor.save_stats(json_path)
        heatmap_evaluations, hole_assignments_to_test = generate_heatmap_complete(pomdp_sketch, fsc)
        with open(os.path.join(project_path, "heatmap_evaluations.txt"), 'a') as f:
            f.write(f"Heatmap evaluations, the new assignment is: {hole_assignment}\n")
            for i, evaluation in enumerate(heatmap_evaluations):
                f.write(f"{hole_assignments_to_test[i]}: {evaluation}\n")
            f.write("\n")
        pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
        last_hole = hole_assignment

        extractor.add_new_pomdp(pomdp,agent)


    if profiling:
        pr.disable()
        stats = pr.create_stats()
        pstats.Stats(stats).sort_stats("tottime").print_stats(10)
    return


main()
