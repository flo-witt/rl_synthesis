from dataclasses import dataclass
from shielding.model_info import ModelInfo
import random
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def clamp_distribution(distribution, allowed_actions):
    """Clamp a distribution to only allowed actions."""
    total_prob = sum(prob for action, prob in enumerate(distribution) if action in allowed_actions)
    if not total_prob > 0:
        # If no safe actions, return a uniform distribution over allowed actions
        uniform_prob = 1.0 / len(allowed_actions)
        return [uniform_prob if action in allowed_actions else 0.0 for action in range(len(distribution))]
    return [prob / total_prob if action in allowed_actions else 0.0 for action, prob in enumerate(distribution)]

class Shield:
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info

        # just for evaluation
        self.blocked_actions = 0

    def correct(self, last_action: any, current_state: any, distribution):
        """Correct a distribution."""
        raise NotImplementedError("Not implemented")
    
    def report_info(self):
        return {}

class IdentityShield(Shield):
    def __init__(self, model_info):
        super().__init__(model_info)
    def correct(self, last_action, current_state, distribution, reset):
        return distribution

class StandardShield(Shield):
    def __init__(self, model_info):
        super().__init__(model_info)

    def _allow_action(self, state, action):
        next_val = 0.0
        row_index = self.model_info.model.transition_matrix.get_row_group_start(state) + action
        row = self.model_info.model.transition_matrix.get_row(row_index)
        for entry in row:
            next_val += entry.value() * self.model_info.vmin[entry.column]
        return next_val <= self.model_info.vmin[state]

    def correct(self, last_action, current_state, distribution, reset):
        actions = self.model_info.model.get_nr_available_actions(current_state)
        allowed_actions = [a for a in range(actions) if self._allow_action(current_state, a)]
        return clamp_distribution(distribution, allowed_actions)

# class PessimisticShield(Shield):
#     def __init__(self, model_info: ModelInfo, nu: float):
#         super().__init__(model_info)
#         self.incurred_safety = 0.0
#         self.path_prob = 1.0
#         self.last_state = None
#         self.bmax = self.model_info.vmax[self.model_info.model.initial_states[0]] - nu
#         self.standard_shield = StandardShield(model_info)
#         self.last_distr = None

#     def _qmax(self, state, distr):
#         qmax = 0.0
#         actions = self.model_info.model.get_choice(state).transition
#         for (actionprob, action) in distr:
#             branch = actions[self.model_info.map_actions(action)]
#             for (value, next_state) in branch:
#                 qmax += actionprob * value * self.model_info.vmax[next_state.id]
#         return qmax

#     def correct(self, last_action, current_state, distribution, reset):
#         actions = self.model_info.model.get_nr_available_actions(current_state)
#         if reset:
#             # Reset shield
#             self.incurred_safety = 0.0
#             self.path_prob = 1.0
#         else:
#             assert self.last_state is not None, "Last state is None on non-reset."
#             assert self.last_distr is not None, "Last distribution is None on non-reset."



#             prob = next_state[0]
#             last_action_prob = [p for p, a in self.last_distr if a == last_action][0]
#             self.path_prob = self.path_prob * prob[0] * last_action_prob


#         qmax = self._qmax(state, distribution)
#         self.incurred_safety += self.path_prob * (self.model_info.vmax[state] - qmax)

#         self.last_state = current_state
#         self.last_distr = distribution

#         if self.incurred_safety > self.bmax:
#             return distribution
#         else:
#             return self.standard_shield.correct(last_action, current_state, distribution)

#     def report_info(self):
#         return {
#             "shield": "pessimistic",
#             "bmax": self.bmax,
#             "safety": self.incurred_safety,
#             "pathprob": self.path_prob,
#             "safety > bmax": self.incurred_safety > self.bmax,
#         }


# class PessimisticShield2(Shield):
#     def __init__(self, model_info: ModelInfo, nu: float):
#         super().__init__(model_info)
#         self.incurred_risk = 0.0
#         self.path_prob = 1.0
#         self.last_state = "init"
#         self.bmin = nu - self.model_info.vmin[0]
#         self.standard_shield = StandardShield(model_info)

#     def _qmax(self, state, distr):
#         qmax = 0.0
#         actions = self.model_info.model.get_choice(state).transition
#         for (actionprob, action) in distr:
#             branch = actions[self.model_info.map_actions(action)]
#             for (value, next_state) in branch:
#                 qmax += actionprob * value * self.model_info.vmax[next_state.id]
#         return qmax

#     def correct(self, last_action, current_state, distribution):
#         state = self.model_info.map_states(current_state)
#         if last_action is None:
#             # Reset shield
#             self.incurred_risk = 0.0
#             self.path_prob = 1.0
#             actions = self.model_info.model.get_choice(self.model_info.model.get_initial_state()).transition
#             assert len(actions) == 1
#             action = list(actions.keys())[0]
#             prob = [x for x in actions[action].branch if x[1].id == state][0]
#             self.path_prob = self.path_prob * prob[0]
#         else:
#             actions = self.model_info.model.get_choice(self.model_info.map_states(self.last_state)).transition
#             probs = actions[self.model_info.map_actions(last_action)]
#             prob = [x for x in probs.branch if x[1].id == state][0]
#             self.path_prob = self.path_prob * prob[0]


#         qmax = self._qmax(state, distribution)
#         self.incurred_risk += self.path_prob * (qmax - self.model_info.vmin[state])

#         self.last_state = current_state

#         if self.incurred_risk <= self.bmin:
#             return distribution
#         else:
#             return self.standard_shield.correct(last_action, current_state, distribution)

#     def report_info(self):
#         return {
#             "shield": "pessimistic2",
#             "bmax": self.bmin,
#             "risk": self.incurred_risk,
#             "pathprob": self.path_prob,
#             "risk <= bmin": self.incurred_risk <= self.bmin,
#         }


# type DistributionFull = list[float]

# @dataclass
# class Node:
#     successors: "dict[tuple[int, int], Node]"
#     predecessor: "Node"
#     last_played_action: int
#     distributions: list[DistributionFull]
#     state_in_mc: int
#     value: float

#     def number_of_tree_nodes(self) -> int:
#         count = 1
#         for succ in self.successors.values():
#             count += succ.number_of_tree_nodes()
#         return count

# class SelfConstructingShield(Shield):
#     # try with tree structure

#     def __init__(self, model_info: ModelInfo, nu: float):
#         super().__init__(model_info)
#         self.nu = nu

#         # assumption
#         initial_state = 0

#         # get vmin
#         self.initial_node = Node({}, None, None, [], initial_state, self.model_info.vmin[initial_state])

#         self.current_node = self.initial_node

#         self.initial_distr = {}
#         self.initialize_init_distr()
#         self.vmin_actions_distributions = []
#         self.vmin_actions = []
#         self.initialize_vmin_actions()

#     # this is maybe not needed but I'm not used to working with stormvogel models :D
#     def initialize_init_distr(self):
#         actions = self.model_info.model.get_choice(0).transition
#         assert len(actions) == 1
#         action = list(actions.keys())[0]
#         for branch_prob, branch_state in actions[action].branch:
#             self.initial_distr[branch_state.id] = branch_prob

#     def initialize_vmin_actions(self):
#         for state in self.model_info.model.states:
#             actions = self.model_info.model.get_choice(state).transition
#             if len(actions) <= 1:
#                 self.vmin_actions_distributions.append([])
#                 self.vmin_actions.append([])
#                 continue
#             vmin_actions_distributions = []
#             vmin_actions = []
#             for action in range(len(actions)):
#                 branch = actions[self.model_info.map_actions(action)]
#                 val = 0.0
#                 for (value, next_state) in branch:
#                     val += value * self.model_info.vmin[next_state.id]
#                 if val <= self.model_info.vmin[state]:
#                     vmin_actions_distributions.append([1.0 if a == action else 0.0 for a in range(len(actions))])
#                     vmin_actions.append(action)
#             self.vmin_actions_distributions.append(vmin_actions_distributions)
#             self.vmin_actions.append(vmin_actions)

#     # this can be optimized probably?
#     def back_propagate_values(self, node: Node):
#         while node is not None:
#             # initial node
#             if node.predecessor is None:
#                 val = 0.0
#                 for state, transition_prob in self.initial_distr.items():
#                     if (state, None) not in node.successors.keys():
#                         val += transition_prob * self.model_info.vmin[state]
#                     else:
#                         val += transition_prob * node.successors[(state, None)].value
#                 node.value = val
#             # every other node
#             else:
#                 # compute value from successors
#                 best_value = float('-inf')
#                 # points of the convex set
#                 all_distributions = node.distributions + self.vmin_actions_distributions[node.state_in_mc]
#                 for distr in all_distributions:
#                     q_value = 0.0
#                     actions = self.model_info.model.get_choice(node.state_in_mc).transition
#                     for action_index, action_prob in enumerate(distr):
#                         if action_prob == 0:
#                             continue
#                         branch = actions[self.model_info.map_actions(action_index)]
#                         for (value, next_state) in branch:
#                             if (next_state.id, action_index) not in node.successors.keys():
#                                 q_value += action_prob * value * self.model_info.vmin[next_state.id]
#                             else:
#                                 q_value += action_prob * value * node.successors[(next_state.id, action_index)].value
#                     if q_value > best_value:
#                         best_value = q_value
#                 assert best_value != float('-inf')
#                 node.value = best_value
#             node = node.predecessor

#     def point_in_convex_hull(self, hull_vertices, point, tolerance=1e-12):
#         vertices = np.array([tuple(v) for v in hull_vertices])
#         p = np.array(point)
#         # Use coordinates directly to check if point is in the convex hull (i.e. the point is linear combination of the basis vectors with non-negative coefficients summing to 1)
#         c = np.zeros(len(vertices))
#         A_eq = np.vstack([vertices.T, np.ones(len(vertices))])
#         b_eq = np.append(p, 1)
#         bounds = [(0, 1)] * len(vertices)
#         res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
#         # Check if the solution coefficients are all within bounds and sum to 1
#         if not (res.success and res.status == 0):
#             return False
#         if np.any(res.x < -tolerance) or np.any(res.x > 1 + tolerance):
#             return False
#         if not np.isclose(np.sum(res.x), 1, atol=tolerance):
#             return False
#         return True

#     def correct(self, last_action, current_state, distribution):

#         current_state_index = self.model_info.map_states(current_state)

#         if last_action is None:
#             # we are looking at a different trace now, so we need to update the values on the previous trace
#             if len(self.initial_node.successors) > 0:
#                 self.back_propagate_values(self.current_node)
#             self.current_node = self.initial_node

#         if (current_state_index, last_action) not in self.current_node.successors.keys():
#             self.current_node.successors[(current_state_index, last_action)] = Node({}, self.current_node, last_action, [], current_state_index, self.model_info.vmin[current_state_index])
#             self.current_node = self.current_node.successors[(current_state_index, last_action)]
#         else:
#             self.current_node = self.current_node.successors[(current_state_index, last_action)]

#         full_distribution = [0.0 for _ in range(len(self.model_info.model.get_choice(current_state_index).transition))]
#         for prob, action in distribution:
#             full_distribution[action] = prob

#         output_distribution = distribution

#         # check if current distribution is inside the convex set
#         if not self.point_in_convex_hull(self.current_node.distributions + self.vmin_actions_distributions[current_state_index], full_distribution):
#             self.current_node.distributions.append(full_distribution)

#             self.back_propagate_values(self.current_node)

#             if self.initial_node.value > self.nu:
#                 self.blocked_actions += 1
#                 self.current_node.distributions.pop()
#                 output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state_index])

#         return output_distribution


# # TODO think about nice implementation where you could easily parameterize the behaviour anywhere between the two version of the self-constructing shield
# class SelfConstructingShieldDistributions(SelfConstructingShield):
#     def __init__(self, model_info: ModelInfo, nu: float):
#         super().__init__(model_info, nu)

#         self.last_distribution_index = None

#     # this can be optimized probably?
#     def back_propagate_values(self, node: Node):
#         while node is not None:
#             # initial node
#             if node.predecessor is None:
#                 val = 0.0
#                 for state, transition_prob in self.initial_distr.items():
#                     if (state, None) not in node.successors.keys():
#                         val += transition_prob * self.model_info.vmin[state]
#                     else:
#                         val += transition_prob * node.successors[(state, None)].value
#                 node.value = val
#             # every other node
#             else:
#                 # compute value from successors
#                 best_value = float('-inf')
#                 # points of the convex set
#                 all_distributions = self.vmin_actions_distributions[node.state_in_mc] + node.distributions
#                 for distr_index, distr in enumerate(all_distributions):
#                     q_value = 0.0
#                     actions = self.model_info.model.get_choice(node.state_in_mc).transition
#                     for action_index, action_prob in enumerate(distr):
#                         if action_prob == 0:
#                             continue
#                         branch = actions[self.model_info.map_actions(action_index)]
#                         for (value, next_state) in branch:
#                             if (next_state.id, distr_index) not in node.successors.keys():
#                                 q_value += action_prob * value * self.model_info.vmin[next_state.id]
#                             else:
#                                 q_value += action_prob * value * node.successors[(next_state.id, distr_index)].value
#                     if q_value > best_value:
#                         best_value = q_value
#                 assert best_value != float('-inf')
#                 node.value = best_value
#             node = node.predecessor

#     def correct(self, last_action, current_state, distribution):

#         current_state_index = self.model_info.map_states(current_state)

#         if last_action is None:
#             # we are looking at a different trace now, so we need to update the values on the previous trace
#             if len(self.initial_node.successors) > 0:
#                 self.back_propagate_values(self.current_node)
#             self.last_distribution_index = None
#             self.current_node = self.initial_node

#         # Change compared to parent class: last_played_action now stores the index of the played distribution
#         if (current_state_index, self.last_distribution_index) not in self.current_node.successors.keys():
#             all_dist = self.vmin_actions_distributions[self.current_node.state_in_mc] + self.current_node.distributions
#             assert self.last_distribution_index is None and len(all_dist) == 0 or self.last_distribution_index < len(all_dist)
#             self.current_node.successors[(current_state_index, self.last_distribution_index)] = Node({}, self.current_node, self.last_distribution_index, [], current_state_index, self.model_info.vmin[current_state_index])
#             self.current_node = self.current_node.successors[(current_state_index, self.last_distribution_index)]
#         else:
#             self.current_node = self.current_node.successors[(current_state_index, self.last_distribution_index)]

#         full_distribution = [0.0 for _ in range(len(self.model_info.model.get_choice(current_state_index).transition))]
#         for prob, action in distribution:
#             full_distribution[action] = prob

#         output_distribution = distribution

#         # check if current distribution is inside the convex set
#         if not self.point_in_convex_hull(self.current_node.distributions + self.vmin_actions_distributions[current_state_index], full_distribution):
#             self.current_node.distributions.append(full_distribution)

#             self.back_propagate_values(self.current_node)

#             if self.initial_node.value > self.nu:
#                 self.blocked_actions += 1
#                 self.current_node.distributions.pop()
#                 output_distribution = clamp_distribution(distribution, self.vmin_actions[current_state_index])

#         # Change compared to parent class: last_played_action now stores the index of the played distribution
#         full_output_distribution = [0.0 for _ in range(len(self.model_info.model.get_choice(current_state_index).transition))]
#         for prob, action in output_distribution:
#             full_output_distribution[action] = prob
#         if full_output_distribution in self.vmin_actions_distributions[current_state_index]:
#             self.last_distribution_index = self.vmin_actions_distributions[current_state_index].index(full_output_distribution)
#         elif full_output_distribution in self.current_node.distributions:
#             self.last_distribution_index = len(self.vmin_actions_distributions[current_state_index]) + self.current_node.distributions.index(full_output_distribution)
#         else:
#             self.current_node.distributions.append(full_output_distribution)
#             self.last_distribution_index = len(self.vmin_actions_distributions[current_state_index]) + len(self.current_node.distributions) - 1

#         return output_distribution