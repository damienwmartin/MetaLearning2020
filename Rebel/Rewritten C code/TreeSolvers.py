import torch
from .subgame_solving_h import ISubgameSolver
from .subgame_solving_cc import *

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
    
    def update_sum_strat(public_node, traverser, br_strategies, traverser_beliefs):

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
    
    def step(traverser):

        br_strategy = self.br_solver.compute_br(traverser, self.average_strategies, self.initial_beliefs, self.root_values[traverser])
        num_update = self.num_strategies / 2 + 1
        alpha = 2 / (num_update + 1) if self.params.linear_update else 1 / num_update
        
        for i in range(len(self.root_values[traverser])):
            root_values_means[traverser][i] +=  (root_values[traverser][i] - root_values_means[traverser][i])*alpha
        
        self.update_sum_strat(0, traverser, br_strategy, initial_beliefs[traverser])

        for node in range(len(self.tree)):
            if self.tree[node].num_children() and tree[node].state.player_id == traverser:
