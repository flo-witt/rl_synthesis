# PAYNT + DRL

This toolkit implements the PAYNT-DRL loop. The implementation contains the modified version of the PAYNT project (original PAYNT description below) and our DRL extension. The sole implementation of DRL is in the rl_src folder, while the most parts of the closed DRL-PAYNT loop is in the ./robust_rl and in the ./paynt/rl_extension.



## Installation
  The installation was tried on PC with Ubuntu 24.04. Before the installation, please install python3.10 (https://askubuntu.com/questions/682869/how-do-i-install-a-different-python-version-using-apt-get), since the official version of TensorFlow Agents currently does not support newer versions of Python. 
  Then run ./install.sh script. 
  
  If you want to install our framework manually, follow those steps: 
 
 Install PAYNT (below).
 Install VecStorm (go to prerequisites/vec_storm) to the same venv.
 Then use following commands:

```shell
source prerequisites/venv/bin/activate
cd ./VecStorm
pip install -e .
cd ..
pip install jax==0.5.3
pip install tensorflow==2.15 ml-dtypes==0.2.0
pip install tf_agents==0.19.0
pip install tqdm dill matplotlib pandas seaborn networkx
pip install aalpy
pip install scikit-learn
cd  ./rl_src
pip install -e .
cd  ..
```


## Usage and the Expected Output

To run all the experiments we mention in the paper, simply run the ./runner_large.sh script, which runs all our experiments with reproducible seeds (always 12345, 23456, 34567, 45678, 56789 in robust setting, 67890, 78901, 89012, 90123, 01234 added in a single POMDP setting). To make it more readable, we separated four similar main experiment loops with different parameters. The results then start generating in the models_robust and models_single_pomdp in json files. The complete experiment takes over 175 hours (each experiment takes 1 hour to complete, the GRU extraction experiments 2 hours).

Running python3.10 robust_pomdps_rl.py with activated venv from prerequisites starts script with simple --help describing all possible parameters. If you run the script with some model that has a family with only a single member, the algorithm will perform long training and a single extraction. If you select some model from the models_robust, the algorithm starts a robust loop. Overall, this script is our main entrypoint to our algorithms and supports running our toolkit for any model.

If you performed the experiment (or any other from our benchmark set), you can get the figures by:
  - To generate the paper data in form of tables, run the installed python environment and script generate_tables.py
  - To generate all the figures in our paper, run generate_convergence_curves.py
  - If you want to generate figures/tables for other data, just change the input directory in the scripts.

If you want to run RL for purely single POMDP setting (without the family of size 1), check the readme in the rl_src directory. 

## Structure
  - models/models_robust -- contains all our benchmark models for the HM POMDP setting.
  - models/models_single_family -- contains all our benchmark models for the Single POMDP setting.
  - paynt/rl_extension/self_interpretable_interface -- calls for both Alergia and SIG extraction methods. 
  - robust_rl/robust_rl_trainer.py -- there is implemented the extraction loop in the extraction_loop() function. The file also contains method extract_fsc(), that calls our extraction methods. 
  - rl_src/environment/environment_wrapper_vec.py -- environment over vectorized simulator. Contains reward function definition that are suggested to adjust modify with new models -- the algorithm runs with any corrent PRISM model with a specification, but it would use default reward function that might not correspond with the proposed task.
  - rl_src/interpreters/networks/fsc_like_actor_network.py contains implementation of the architecture of our Gumbel-Softmax and other self-interpretable options not mentioned in the paper. The interpreters folder itself contains the implementation of behavioral cloning and all the stuff necessary for the extraction.
  - rl_src/agents/recurrent_ppo_agent.py -- contains implementation of our PPO agent.

## Framework and Sources
The implementation is primarily based on PAYNT with Stormpy, TensorFlow Agents framework, which implements many important blocks of this project as reinforcement learning algorithms, and the AALpy Automata learning library (https://github.com/DES-Lab/AALpy). We also took some inspiration and in case of .rl_src/environment/pomdp_builder, we took the code from repository: [Shielding](https://github.com/stevencarrau/safe_RL_POMDPs).
```

## Credits

- Main author: David Hudák (ihudak@fit.vutbr.cz)
- PAYNT: Roman Andriushchenko (https://github.com/randriu) and Filip Macák (https://github.com/TheGreatfpmK)
- VecStorm: Modified version of the implementation by Martin Kurečka (https://github.com/kurecka/VecStorm)
- Automata learning implementation: Martin Tappler (https://github.com/mtappler)
- Technical and theoretical consultations: Maris F.L. Galesloot (maris.galesloot@ru.nl)
- Supervisor: Milan Češka (ceskam@fit.vutbr.cz)
