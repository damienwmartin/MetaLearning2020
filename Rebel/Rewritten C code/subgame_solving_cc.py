from .util_h import *

def compute_reach_probabilities(tree, strategy, initial_beliefs, player, reach_probabilities):
    """
    Recomputes the probability that a certain node is reached
    """
    num_hands = len(initial_beliefs)
    for node_id in range(len(tree)):
        if node_id:
            node = tree[node_id]
            state = node.state
            last_action_player_id = tree[node.parent].state.player_id
            last_action = state.last_bid

            if player == last_action_player_id:
                for hand in range(game.num_hands()):
                    reach_probabilities[node_id][hand] = reach_probabilities[node.parent][hand]*strategy[node.parent][hand][last_action]
            else:
                reach_probabilities[node_id] = reach_probabilities[node.parent]

        else:
            reach_probabilities[node_id]= initial_beliefs



def compute_expected_terminal_values(game, last_bid, inverse, op_reach_probabilities):
    
    inv = 2*int(inverse) - 1
    values = compute_win_probability(game, last_bid, op_reach_probabilities)
    belief_sum = sum(op_reach_probabilities)

    for i in range(len(values)):
        values[i] = (2*values[i] - belief_sum)*inv
    
    return values


def get_ery_size(game):

    return 1 + 1 + game.num_actions() + game.num_hands()*2


def get_uniform_reach_weighted_strategy(game, tree, initial_beliefs):

    strategy = get_uniform_strategy(game, tree)
    reach_probabilities_buffer = [[0 for i in range(tree.size())] for j in range(game.num_hands())]
    
    for traverser in [0, 1]:
        compute_reach_probabilities(tree, strategy, initial_beliefs[traverser], traverser, reach_probabilities_buffer)
        for node in range(len(tree)):
            if tree[node].num_children() and tree_node.state.player_id == traverser:
                action_begin, action_end = game.get_bid_range(tree[node].state)
                for hand in range(game.num_hands()):
                    for action in range(action_begin, action_end):
                        strategy[node][hand][action] *= reach_probabilities_buffer[node][hand]
    
    return strategy