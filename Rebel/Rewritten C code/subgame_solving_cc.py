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


def get_query_size(game):

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

def compute_win_probability(game, action, beliefs):
    unpacked_action = game.unpack_action(action)
    believed_counts = [0 for i in range(game.total_num_dice() + 1)]
    for hand in range(len(beliefs)):
        matches = game.num_matches(hand, unpacked_bet[1])
        believed_counts[matches] += beliefs[hand]
    
    for i in range(1, len(believed_counts)):
        believed_counts[-1-i] += believed_counts[-i]
    
    values = []
    for hand in range(len(beliefs)):
        matches = game.num_matches(hand, unpacked_action[1])
        left_to_win = max(0, unpacked_action[0] - matches)
        prob_to_win = believed_counts[left_to_win]
        values.append(prob_to_win)
    
    return values

def build_solver(game, root, beliefs, params, net):

    if params.use_cfr:
        return CFR(game, root, net, beliefs, params)
    else:
        return FP(game, root, net, beliefs, params)
