# using Paynt for POMDP sketches

import paynt.cli
import paynt.parser.sketch
import paynt.quotient.pomdp_family
import paynt.quotient.fsc
import paynt.synthesizer.synthesizer_onebyone
import paynt.synthesizer.synthesizer_ar

import payntbind

import os
import random
import cProfile, pstats
import itertools

def load_sketch(project_path):
    project_path = os.path.abspath(project_path)
    sketch_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")    
    pomdp_sketch = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return pomdp_sketch

def assignment_to_pomdp(pomdp_sketch, assignment):
    pomdp = pomdp_sketch.build_pomdp(assignment).model
    updated = payntbind.synthesis.restoreActionsInAbsorbingStates(pomdp)
    if updated is not None: pomdp = updated
    action_labels,_ = payntbind.synthesis.extractActionLabels(pomdp);
    num_actions = len(action_labels)
    pomdp,choice_to_true_action = payntbind.synthesis.enableAllActions(pomdp)
    observation_action_to_true_action = [None]* pomdp.nr_observations
    for state in range(pomdp.nr_states):
        obs = pomdp.observations[state]
        if observation_action_to_true_action[obs] is not None:
            continue
        observation_action_to_true_action[obs] = [None] * num_actions
        choice_0 = pomdp.transition_matrix.get_row_group_start(state)
        for action in range(num_actions):
            choice = choice_0+action
            true_action = choice_to_true_action[choice]
            observation_action_to_true_action[obs][action] = true_action
    return pomdp,observation_action_to_true_action


def random_fsc(pomdp_sketch, num_nodes):
    num_obs = pomdp_sketch.num_observations
    fsc = paynt.quotient.fsc.Fsc(num_nodes, num_obs)
    for n in range(num_nodes):
        for z in range(num_obs):
            actions = pomdp_sketch.observation_to_actions[z]
            updates = list(range(num_nodes))
            transitions = list(itertools.product(actions, updates))
            fsc.transitions[n][z] = { tr:1/len(transitions) for tr in transitions }
    print("FSC transitions:", fsc.transitions)
    return fsc

def random_fsc_factored(pomdp_sketch, num_nodes):
    num_obs = pomdp_sketch.num_observations
    fsc = paynt.quotient.fsc.FscFactored(num_nodes, num_obs)
    # action function if of type NxZ -> Distr(Act)
    for n in range(num_nodes):
        for z in range(num_obs):
            actions = pomdp_sketch.observation_to_actions[z]
            fsc.action_function[n][z] = { action:1/len(actions) for action in actions }
    # memory update function is of type NxZ -> Distr(N) and is posterior-aware
    # note: this is currently inconsistent with definitions in paynt.quotient.fsc.FSC, but let's see how this works
    for n in range(num_nodes):
        for z in range(num_obs):
            fsc.update_function[n][z] = { n_new:1/num_nodes for n_new in range(num_nodes) }
    return fsc


def main():
    profiling = True
    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    paynt.utils.timer.GlobalTimer.start()

    # random.seed(11)

    # enable PAYNT logging
    paynt.cli.setup_logger()

    # load sketch
    # project_path="models/pomdp/sketches/avoid"
    project_path="models/pomdp/sketches/obstacles-10-2"
    # project_path="models/pomdp/sketches/obstacles-4-1"
    pomdp_sketch = load_sketch(project_path)

    # construct POMDP from the given hole assignment
    hole_assignment = pomdp_sketch.family.pick_any()
    pomdp,observation_action_to_true_action = assignment_to_pomdp(pomdp_sketch,hole_assignment)

    # construct an arbitrary 3-FSC
    # fsc = random_fsc_factored(pomdp_sketch,3)
    fsc = random_fsc(pomdp_sketch,3)
    
    # unfold this FSC into the quotient POMDP to obtain the quotient MDP (DTMC sketch)
    # negate specification since we are interested in violating/worst assignments
    dtmc_sketch =  pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)

    # solve
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
    synthesizer.run()


    if profiling:
        profiler.disable()
        stats = profiler.create_stats()
        pstats.Stats(profiler).sort_stats('tottime').print_stats(10)


if __name__ == "__main__":
    main()
