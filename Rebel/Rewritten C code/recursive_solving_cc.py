import pyspiel
import collections
from .util_h import normalize_probabilities_safe

EPSILON = 1e-20

def normalize_beliefs_inplace(beliefs):
    return normalize_probabilities_safe(beliefs, EPSILON)

def compute_strategy_recursive(game, tree, node_id, beliefs, solver_builder, strategy):
    """
    The function computes the strategy for the node and for all its children

    Parameters:
    
    solver_builder: A function that takes in parameters and outputs a solver of class IGameSolver (see subgame_solver_h.py)
    """
    node = tree[node_id]
    state = node.state

    if state.is_terminal():
        return None
    
    else:
        solver = solver_builder(game, node_id, state, beliefs)
        solver.multistep()
        strategy.at(node_id) = solver.get_strategy()[0] # Change definition of strategy

        # Note - will find out what a belief is and what a strategy is in terms of code
        for child_node_id in range(node.children_begin, node.children_end)
            new_beliefs = beliefs
            action = state.legal_actions()[child_node_id - node.children_begin]

            normalize_beliefs_inplace(new_beliefs[state.player_id])
            compute_strategy_recursive(game, tree, child_node_id, new_beliefs, solver_builder, strategy)


def compute_strategy_recursive_to_leaf(game, tree, node_id, beliefs, solver_builder, use_sampling_strategy, strategy)

    node = tree[node_id]
    state = node.state
    if state.is_terminal()
        return None
    
    solver = solver_builder(game, node_id, state, beliefs)
    solver.multistep()

    # The traversal queue stores tuples of (full_node_id, partial_node_id, unnormalized_beliefs at the node)
    traversal_queue = collections.deque()
    traversal_queue.append((node_id, 0, beliefs))

    partial_strategy = (solver.get_sampling_strategy() if use_sampling_strategy else solver.get_strategy())
    partial_belief_strategy = (solver.get_belief_propagation_strategy() if use_sampling_strategy else solver.get_strategy())
    partial_tree = solver.get_tree()

    while len(traversal_queue) != 0:
        full_node_id, partial_node_id, node_reaches = traversal_queue.popleft()
        strategy[full_node_id] = partial_strategy[partial_node_id]

        full_node = tree[full_node_id]
        partial_node = partial_tree[partial_node_id]
        
        for i in range(partial_node.num_children()):
            child_reaches = node_reaches
            pid = full_node.state.player_id
            action = partial_node.state.legal_actions()[i]

            ### Something Liar's Dice Specific Included? ###

            traversal_queue.append(full_node.children_begin + i, partial_node.children_begin + i, child_reaches)