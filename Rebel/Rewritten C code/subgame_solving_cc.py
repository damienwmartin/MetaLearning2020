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

def compute_exploitability2(game, strategy):
    root = game.get_initial_state()
    tree = unroll_tree(game, root, 1000000)
    beliefs = ([1/game.num_hands() for i in range(game.num_hands())], [1/game.num_hands() for i in range(game.num_hands())])
    solver = BRSolver(game, tree, None)
    values0 = solver.compute_br(0, strategy, beliefs)
    values1 = solver.compute_br(1, strategy, beliefs)

    return (sum(values0)/len(values0), sum(values1)/len(values1))

def compute_exploitability(game, strategy):
    exploitabilities = compute_exploitability2(game, strategy)
    return (exploitabilities[0] + exploitabilities[1])/2

def compute_strategy_stats(game, strategy):
    uniform_beliefs = get_initial_beliefs(game)[0]
    tree = unroll_tree(game)
    stats = TreeStrategyStats(tree=tree)

    reach_probabilities = stats.reach_probabilities

    #TODO: Finish this function!

def compute_ev(game, strategy1, strategy2):
    tree = unroll_tree(game)
    op_reach_probabilities = [[0 for i in range(game.num_hands())] for j in range(len(tree))]
    values = [[] for i in range(len(tree))]
    player = 0
    compute_reach_probabilities(tree, strategy2, get_initial_beliefs(game)[0], 1 - player, op_reach_probabilities)

    for node_id in range(len(tree)):
        node = tree[node_id]
        state = node.state
        if not node.num_children():
            assert game.is_terminal(state)
            last_bid = tree[node.parent].state.last_bid
            values[node_id] = compute_expected_terminal_values(game, last_bid, state.player_id != player, op_reach_probabilities[node_id])
        elif state.player_id == player:
            values[node_id] = resize(values[node_id], game.num_hands())
            for child_node_id, action in ChildrenActionIt(node, game):
                values[node_id][hand] += (strategy1[node_id][hand][action] * values[child_node_id][hand])
        else:
            values[node_id] = resize(values[node_id], game.num_hands())
            for child_node_id in ChildrenIt(node):
                for hand in range(game.num_hands()):
                    values[node_id][hand] += values[child_node_id][hand]
    
    return values[0]

def compute_ev2(game, strategy1, strategy2):

    ev1 = sum(compute_ev(game, strategy1, strategy2))/game.num_hands()
    ev2 = -1*sum(compute_ev(game, strategy2, strategy1))/game.num_hands()

    return (ev1, ev2)

def compute_immediate_regrets(game, strategies):

    from .TreeSolvers import PartialTreeTraverser
    
    tree = unroll_tree(game):
    regrets = [[[0 for k in range(game.num_actions())] for j in range(game.num_hands())] for i in range(len(tree))]
    tree_traverser = PartialTreeTraverser(game, tree, None)
    initial_beliefs = get_initial_beliefs(game)[0]
    for strategy_id in range(len(strategies)):
        last_strategies = strategies[strategy_id]
        tree_traverser.precompute_reaches(last_strategies, initial_beliefs, 0)
        tree_traverser.precompute_reaches(last_strategies, initial_beliefs, 1)
        for traverser in range(2):
            tree_traverser.precompute_all_leaf_values(traverser)
            for public_node_id in range(len(tree)):
                node = tree[public_node_id]
                if node.num_children():
                    state = node.state
                    value = [0 for i in range(len(tree_traverser.traverser_values[public_node_id]))]
                    if state.player_id == traverser:
                        for child_node, action in ChildrenActionIt(node, game):
                            action_value = tree_traverser.traverser_values[child_node]
                            for hand in range(game.num_hands()):
                                regrets[public_node_id][hand][action] += action_value[hand]
                                value[hand] += (action_value[hand] * last_strategies[public_node_id][hand][action])
                        for hand in range(game.num_hands()):
                            for child_node, action in ChildrenActionIt(node, game):
                                regrets[public_node_id][hand][action] -= value[hand]
                    else:
                        assert state.player_id == 1 - traverser
                        for child_node in ChildrenIt(node):
                            action_value = tree_traverser.traverser_values[child_node]
                            for hand in range(game.num_hands()):
                                value[hand] += action_value[hand]
                    tree_traverser.traverser_values[public_node_id] = value

    #TODO: finish this function  