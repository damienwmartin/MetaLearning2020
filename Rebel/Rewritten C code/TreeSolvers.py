import torch
from .subgame_solving_h import ISubgameSolver
from .subgame_solving_cc import *
from .util_h import *

class PartialTreeTraverser:

    def __init__(self, game, tree, value_net):

        self.game = game
        self.tree = tree
        self.reach_probabilities = None
        self.traverser_values = None
        self.query_size = get_query_size(game)
        self.output_size = game.num_hands()

        self.pseudo_leaves_indices = []
        self.terminal_indices = []
        self.value_net = value_net
        self.leaf_values = None
        self.net_query_buffer = []

        if value_net:
            for node_id in range(len(tree)):
                node = tree[node_id]
                state = node.state
                if not node.num_children() and not game.is_terminal(state):
                    self.pseudo_leaves_indices.append(node_id)
        else:
            for node in tree:
                if not node.num_children() and not game.is_terminal(state):
                    raise Exception("Found a node that is a non-final leaf. Either provide a value net or increase max_depth.")

        for node_id in range(len(tree)):
            if game.is_terminal(tree[node_id].state):
                self.terminal_indices.append(node_id)
        
        self.leaf_values = torch.empty(len(pseudo_leaves_indices), output_size)
        self.traverser_values = [[0 for j in range(game.num_hands())] for i in range(len(tree))]
        self.reach_probabilities = ([[0 for j in range(game.num_hands())] for i in range(len(tree))], [[0 for j in range(len(tree))] for i in range(game.num_hands())])
    
    def add_training_example(traverser, values):
        query_tensor = torch.zeros(1, self.query_size)
        value_tensor = torch.zeros(1, self.output_size)
        pass

    def precompute_reaches(strategy, initial_beliefs, player):

        compute_reach_probabilities(tree, strategy, initial_beliefs, player, reach_probabilities[player])
    

    def precompute_all_leaf_values(traverser):
        self.query_value_net(traverser)
        self.populate_leaf_values()
        self.precompute_terminal_leaves_values()

    def precompute_terminal_leaves_values(traverser):

        for node_id in self.terminal_indices:
            last_bid = self.tree[self.tree[node_id].parent].state.last_bid
            traverser_values[node_id] = compute_expected_terminal_values(game, last_bid, tree[node_id].state.player_id != traverser, reach_probabilities[1 - traverser][node_id])
    

    def populate_leaf_values():
        if self.pseudo_leaves_indices != []:
            result_acc = self.leaf_values[2] #TODO: WTF is torch::tensor.accessor<float, 2>
            for row in range(len(self.pseudo_leaves_indices)):
                node_id = pseudo_leaves_indices[row]
                for i in range(self.output_size):
                    self.traverser_values[node_id][i] = result_acc[row][i]
    

class BRSolver(PartialTreeTraverser):

    def __init__(self, game, tree, value_net):
        super().__init__(game, tree, value_net)
        self.br_strategies = [[[0 for i in range(game.num_actions())] for j in range(game.num_hands())] for k in range(len(tree))]

        self.num_strategies = 0
        self.params = params
        self.root_values = None
        self.root_values_means = None

    def compute_br(traverser, opponent_strategy, initial_beliefs, values):
        self.precompute_reaches(opponent_strategy, initial_beliefs)
        self.precompute_all_leaf_values(traverser)

        for public_node_id in range(len(tree)):
            node = self.tree[public_node_id]
            value = self.traverser_values[public_node_id]
            
            if node.num_children():
                state = node.state
                value = [0]*len(value)
                if state.player_id == traverser:
                    best_action = [0 for i in range(self.game.num_hands())]
                    for child_node, action in ChildrenActionIt(node, game):
                        new_value = self.traverser_values[child_node]
                        for hand in range(self.game.num_hands()):
                            if child_node == node.children_begin or new_value[hand] > value[hand]:
                                value[hand] = new_value[hand]
                                best_action[hand] = action
                    for hand in range(self.game.num_hands()):
                        self.br_strategies[public_node_id][hand] = [0 for i in range(self.game.num_actions())]
                        self.br_strategies[public_node_id][hand][best_action[hand]] = 1
                else:
                    for child_node in ChildrenIt(node):
                        new_value = traverser_values[child_node]
                        for hand in range(self.game.num_hands()):
                            value[hand] += new_value[hand]
        
        values = self.traverser_values[0] #TODO: Value points to somewhere. Find out where
        return self.br_strategies
    

class FP(ISubgameSolver):

    def __init__(self, game, tree, value_net, beliefs, params):
        self.params = params
        self.game = game
        self.num_strategies = 0
        self.initial_beliefs = beliefs
        self.tree = tree
        self.br_solver = BRSolver(game, tree, value_net)

        self.average_strategies = get_uniform_strategy(game, tree)
        self.last_strategies = average_strategies
        self.sum_strategies = get_uniform_reach_weighted_strategy(game, tree, initial_beliefs)
    
    def update_sum_strat(self, public_node, traverser, br_strategies, traverser_beliefs):

        node = self.tree[public_node]
        state = node.state

        if node.num_children():
            if state.player_id == traverser:
                
                for child_node, action in ChildrenActionIt(node, game):
                    for hand in range(self.game.num_hands()):
                        new_beliefs.append(traverser_beliefs[hand] * br_strategies[public_node][hand][action])
                    
                    self.update_sum_strat(child_node, traverser, br_strategies, new_beliefs)
            else:
                assert state.player_id == 1 - traverser
                for child_node in ChildrenIt(node):
                    self.update_sum_strat(child_node, traverser, br_strategies, traverser_beliefs)
    
    def step(self, traverser):

        br_strategy = self.br_solver.compute_br(traverser, self.average_strategies, self.initial_beliefs, self.root_values[traverser])
        num_update = self.num_strategies / 2 + 1
        alpha = 2 / (num_update + 1) if self.params.linear_update else 1 / num_update
        
        for i in range(len(self.root_values[traverser])):
            root_values_means[traverser][i] +=  (root_values[traverser][i] - root_values_means[traverser][i])*alpha
        
        self.update_sum_strat(0, traverser, br_strategy, initial_beliefs[traverser])

        for node in range(len(self.tree)):
            if self.tree[node].num_children() and tree[node].state.player_id == traverser:
                for hand in range(self.game.num_hands()):
                    if params.linear_update:
                        for j in range(len(self.sum_strategies[node][i])):
                            self.sum_strategies[node][i][j] *= (num_update + 1)/(num_update + 2)
                    """
                    if params.optimistic:
                        normalize_probabilities_safe(self.sum_strategies[node][i], self.last_strategies[node][i], self.average_strategies[node][i])
                    else:
                        normalize_probabilities_safe(self.sum_strategies[node][i], self.average_strategies[node][i])
                    """
        
        self.num_strategies += 1
    
    def multistep(self):
        for i in range(params.num_iter):
            self.step(i % 2)
    
    def update_value_network(self)
        br_solver.add_training_example(0, get_hand_values(0))
        br_solver.add_training_example(1, get_hand_values(1))
    
    def get_strategy(self):
        return self.average_strategies
    
    def get_tree(self):
        return self.tree


class CFR(ISubgameSolver, PartialTreeTraverser):

    def __init__(self, game, tree, value_net, beliefs, params):
        super(PartialTreeTraverser, self).__init__(game, tree, value_net)
        self.params = params
        self.initial_beliefs = beliefs
        self.num_steps = [0, 0]
        
        self.average_strategies = get_uniform_strategy(game, tree)
        self.last_strategies = self.average_strategies
        self.sum_strategies = get_uniform_reach_weighted_strategy(game, tree, initial_beliefs)
        self.regrets = [[[0 for i in range(game.num_actions())] for j in range(game.num_hands())] for k in range(len(tree))]
        self.reach_probabilities_buffer = [[0 for in range(game.num_hands())] for j in range(len(tree))]

        self.root_values = [None, None]
        self.root_values_means = [None, None]
    
    def update_regrets(self, traverser):
        self.precompute_reaches(self.last_strategies, self.initial_beliefs)
        self.precompute_all_leaf_values(traverser)

        for public_node_id in range(len(self.tree)):
            node = self.tree[public_node_id]
            if node.num_children():
                state = node.state
                value = [0 for i in range(len(self.traverser_values[public_node_id]))]
                if state.player_id == traverser:
                    for child_node, action in ChildrenActionIt(node, game):
                        action_value = self.traverser_values[child_node]
                        for hand in range(self.game.num_hands()):
                            regrets[public_node_id][hand][action] += action_value[hand]
                            value[hand] += action_value[hand]*last_strategies[public_node_id][hand][action]
                    for hand in range(self.game.num_hands()):
                        for child_node, action in ChildrenActionIt(node, game):
                            regrets[public_node_id][hand][action] -= value[hand]
            else:
                assert state.player_id == 1 - traverser
                for child_node in ChildrenIt(node):
                    action_value = traverser_values[child_node]
                    for hand in range(self.game.num_hands()):
                        value[hand] += action_value[hand]
                
                self.traverser_values[public_node_id] = value
    
    def step(self, traverser):
        self.update_regrets(traverser)
        self.root_values[traverser] = self.traverser_values[0]

        alpha = 2 /(self.num_steps[traverser] + 2) if self.params.linear_update else 1 / (self.num_steps[traverser] + 1)
        self.root_values_means[traverser] = resize(self.root_values_means(traverser), len(self.root_values[traverser]))
        for i in range(len(self.root_values[traverser])):
            self.root_values_means[traverser][i] += (self.root_values[traverser][i] - self.root_values_means[traverser][i])*alpha

        pos_discount = 1
        neg_discount = 1
        strat_discount = 1

        num_strategies = self.num_steps[traverser] + 1
        if self.params.linear_update:
            pos_discount = num_strategies / (num_strategies + 1)
            neg_discount = pos_discount
            strat_discount = pos_discount
        elif self.params.dcfr:
            if self.params.dcfr_alpha < 5:
                pos_discount = num_strategies**self.params.dcfr_alpha / (num_strategies**self.params.dcfr_alpha + 1)
            if self.params.dcfr_beta > -5:
                neg_discount = num_strategies**self.params.dcfr_beta / (num_strategies**self.params.dcfr_beta + 1)
            strat_discount = (num_strategies / (num_strategies + 1))**self.params.dcfr_gamma
        
        for node_id in range(len(self.tree)):
            if self.tree[node_id].num_children() and self.tree[node_id].state.player_id == traverser:
                start, end = self.game.get_bid_ranges(self.tree[node_id].state)
                for hand in range(self.game.num_hands())
                    for action in range(start, end):
                        self.last_strategies[node_id][hand][action] = max(regrets[node_id][hand][action], EPSILON)
                    last_strategies[node][hand] = normalize_probabilities_safe(last_strategies[node][hand])
        
        compute_reach_probabilities(self.tree, self.last_strategies, self.initial_beliefs[traverser], traverser, self.reach_probabilities_buffer)

        for node_id in range(len(self.tree)):
            if self.tree[node_id].num_children() and self.tree[node_id].state.player_id == traverser:
                action_begin, action_end = self.game.get_bid_ranges(self.tree[node_id].state)
                for hand in range(self.game.num_hands()):
                    for action in range(action_begin, action_end):
                        self.regrets[node_id][hand][action] *= (pos_discount if self.regrets[node][i][a] > 0 else neg_discount)
                        self.sum_strategies[node_id][hand][action] *= strat_discount
                        self.sum_strategies[node_id][hand][action] += (self.reach_probabilities_buffer[node_id][hand] * self.last_strategies[node_id][hand][action])
                        self.average_strategies[node_id][hand] = normalize_probabilities_safe(sum_strategies[node_id][hand])
        
        self.num_steps[traverser] += 1
    
    def multistep(self, traverser):
        for i in range(self.params.num_iters):
            self.step(i % 2)
    
    def update_value_network(self):
        self.add_training_example(0, self.get_hand_values(0))
        self.add_training_example(1, self.get_hand_values(1))
    
    def get_strategy(self):
        return self.average_strategies

    def get_sampling_strategy(self):
        return self.last_strategies
    
    def get_belief_propagation_strategies(self):
        return self.last_strategies
    
    def print_strategy(self):
        return "Needs to be implemented :)"
    
    def get_hand_values(self, player_id):
        return self.root_values_means[player_id]