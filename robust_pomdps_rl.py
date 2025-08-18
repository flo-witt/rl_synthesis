
from robust_rl.robust_rl_trainer import RobustTrainer, initialize_extractor
from robust_rl.robust_rl_tools import parse_args
from robust_rl.robust_rl_tools import generate_heatmap_complete, assignment_to_pomdp, create_json_file_name, load_sketch
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


def main():
    args_cmd = parse_args()

    paynt.utils.timer.GlobalTimer.start()

    profiling = True
    if profiling:
        pr = cProfile.Profile()
        pr.enable()

    paynt.cli.setup_logger()

    project_path = args_cmd.project_path
    pomdp_sketch = load_sketch(project_path)
    json_path = create_json_file_name(project_path)


    num_samples_learn = 802
    nr_pomdps = 4

    # This can be useful for extraction and some other stuff.
    family_quotient_numpy = FamilyQuotientNumpy(pomdp_sketch)

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
    args_emulated.max_steps = 601
    args_emulated.geometric_batched_vec_storm = args_cmd.geometric_batched_vec_storm
    args_emulated.without_extraction = args_cmd.without_extraction
    args_emulated.periodic_restarts = args_cmd.periodic_restarts
    args_emulated.noisy_observations = args_cmd.noisy_observations
    args_emulated.shrink_and_perturb = args_cmd.shrink_and_perturb
    args_emulated.shrink_and_perturb_externally = args_cmd.shrink_and_perturb_externally
    # pomdp = initialize_prism_model(prism_path, properties_path, constants="")

    hole_assignment = pomdp_sketch.family.pick_any()
    pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)

    extractor = initialize_extractor(
        pomdp_sketch, args_emulated, family_quotient_numpy)

    agent = extractor.generate_agent(pomdp, args_emulated)
    last_hole = None
    for i in range(nr_pomdps):
        hole_assignment = pomdp_sketch.family.pick_random()
    extractor.extraction_loop(pomdp_sketch, project_path=project_path,
                              nr_initial_pomdps=nr_pomdps, num_samples_learn=num_samples_learn)
    return


main()
