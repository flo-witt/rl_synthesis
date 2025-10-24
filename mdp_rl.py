from tests.general_test_tools import init_args

from robust_rl.robust_rl_tools import load_sketch

import os

from environment.environment_wrapper_vec import EnvironmentWrapperVec
from environment.tf_py_environment import TFPyEnvironment
from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent

from paynt.parser.sketch import Sketch


def load_sketch(project_path):
    project_path = os.path.abspath(project_path)
    sketch_path = os.path.join(project_path, "model.prism")
    properties_path = os.path.join(project_path, "discounted.props")
    pomdp_sketch = Sketch.load_sketch(
        sketch_path, properties_path)
    return pomdp_sketch

def create_json_file_name(project_path, seed = ""):
    """
    Creates a JSON file name based on the project path.
    """
    json_path = os.path.join(project_path, f"benchmark_stats_{seed}.json")
    if os.path.exists(json_path):
        index = 0
        while os.path.exists(os.path.join(project_path, f"benchmark_stats_{seed}_{index}.json")):
            index += 1
        json_path = os.path.join(project_path, f"benchmark_stats_{seed}_{index}.json")
    return json_path


def main():
    project_path = "models_mdp/maze-7"
    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    args = init_args(prism_path=prism_path, properties_path=properties_path, use_rnn_less=True)
    sketch = load_sketch(project_path=project_path)
    
    # ---------------------------------------------------------
    # This is the learning
    environment = EnvironmentWrapperVec(sketch.quotient_mdp, args, num_envs=args.num_environments, enforce_compilation=True)
    tf_env = TFPyEnvironment(environment)
    agent = Recurrent_PPO_agent(
       environment=environment, tf_environment=tf_env, args=args)
    agent.train_agent(iterations=500)
    # ---------------------------------------------------------
    # Save the results

    json_path = create_json_file_name(project_path, seed=args.seed)
    agent.evaluation_result.save_to_json(json_path, new_pomdp=False)
    
    


if __name__ == "__main__":
    main()