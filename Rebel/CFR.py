import numpy as np
import torch
from .game_tree import node_to_number

EPSILON = 1e-100

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
                    # Pseudo-leaves are nodes which are leaves in our depth-limited subgame but not actually terminal states in the game
                    self.pseudo_leaves_indices.append(node_id)
        else:
            for node in tree:
                if not node.num_children() and not game.is_terminal(state):
                    raise Exception("Found a node that is a non-final leaf. Either provide a value net or increase max_depth.")

        for node_id in range(len(tree)):
            if game.is_terminal(tree[node_id].state):
                self.terminal_indices.append(node_id)
        
        self.leaf_values = torch.empty(len(pseudo_leaves_indices), output_size)
        self.traverser_values = np.zeros(len(tree), game.num_hands)
        self.reach_probabilities = (np.zeros(len(tree), game.num_hands), np.zeros(len(tree), game.num_hands))
    
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


class CFR(PartialTreeTraverser):
    """
    Implementation of CFR that was directly translated from the ReBeL repo
    """

    def __init__(self, game, tree, value_net, beliefs, params):
        super().__init__(game, tree, value_net)
        self.params = params
        self.initial_beliefs = beliefs
        self.num_steps = [0, 0]
        
        self.average_strategies = get_uniform_strategy(game, tree)
        self.last_strategies = self.average_strategies
        self.sum_strategies = get_uniform_reach_weighted_strategy(game, tree, initial_beliefs)
        self.regrets = np.zeros((len(tree), game.num_hands, game.num_actions)
        self.reach_probabilities_buffer = np.zeros((len(tree), game.num_hands)

        self.root_values = [None, None]
        self.root_values_means = [None, None]
    
    def update_regrets(self, traverser):
        """
        Computes the regrets associated with a traverser and stores the result in self.regrets
        """
        self.precompute_reaches(self.last_strategies, self.initial_beliefs)
        self.precompute_all_leaf_values(traverser)

        for public_node in self.tree.nodes:
            public_node_id = node_to_number(public_node)
            if public_node['subgame_terminal']:
                state = node['state']
                value = np.zeros_like(self.traverser_values[public_node_id])
                action_values = np.array([self.traverser_values[child_node_id] if child_node_id else [0]*game.num_hands for action, child_node_id in self.game.iter_at_node(public_node_id)) # Need to change way to iterate
                if state.player_id == traverser:
                    regrets[public_node_id] += np.transpose(action_values)
                    value += np.sum(action_value * np.transpose(self.last_strategies[public_node_id]), axis=0)
                    regrets[public_node_id] -= np.vstack([value]*game.num_actions, 1) # Change to 0 for invalid actions
                    
                else:
                    assert state.player_id == 1 - traverser
                    value += action_values
            
                self.traverser_values[public_node_id] = value
                    

"""
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
"""
    
    def step(self, traverser):
        """
        Does a step of the CFR algorithm, and updates the average policies
        """
        self.update_regrets(traverser)
        self.root_values[traverser] = self.traverser_values[0]

        # Updates the average using a factor of alpha, which depends on if we use LCFR or normal CFR
        alpha = 2 /(self.num_steps[traverser] + 2) if self.params.linear_update else 1 / (self.num_steps[traverser] + 1)
        self.root_values_means[traverser] = resize(self.root_values_means(traverser), len(self.root_values[traverser]))

        self.root_values_means[traverser] += alpha*(self.root_values[traverser] - self.root_values_means[traverser])

        """
        for i in range(len(self.root_values[traverser])):
            self.root_values_means[traverser][i] += (self.root_values[traverser][i] - self.root_values_means[traverser][i])*alpha
        """

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
        
        """
        for node_id in range(len(self.tree)):
            if self.tree[node_id].num_children() and self.tree[node_id].state.player_id == traverser:
                start, end = self.game.get_bid_ranges(self.tree[node_id].state)
                for hand in range(self.game.num_hands())
                    for action in range(start, end):
                        self.last_strategies[node_id][hand][action] = max(regrets[node_id][hand][action], EPSILON)
                    last_strategies[node][hand] = normalize_probabilities_safe(last_strategies[node][hand])
        """

        for node in self.tree.nodes:
            state = self.game.node_to_state(node['id'])
            if state[1] == traverser and not node['terminal']:
                start, end = self.game.get_bid_ranges(state)
                self.last_strategies[node_id] = np.max(regrets[node_id], EPSILON)
                for hand in range(self.game.num_hands):
                    self.last_strategies[node_id][hand] = normalize_probabilities_safe(self.last_strategies[node_id][hand])

        compute_reach_probabilities(self.tree, self.last_strategies, self.initial_beliefs[traverser], traverser, self.reach_probabilities_buffer)

        """
        for node_id in range(len(self.tree)):
            if self.tree[node_id].num_children() and self.tree[node_id].state.player_id == traverser:
                action_begin, action_end = self.game.get_bid_ranges(self.tree[node_id].state)
                for hand in range(self.game.num_hands()):
                    for action in range(action_begin, action_end):
                        self.regrets[node_id][hand][action] *= (pos_discount if self.regrets[node][i][a] > 0 else neg_discount)
                        self.sum_strategies[node_id][hand][action] *= strat_discount
                        self.sum_strategies[node_id][hand][action] += (self.reach_probabilities_buffer[node_id][hand] * self.last_strategies[node_id][hand][action])
                        self.average_strategies[node_id][hand] = normalize_probabilities_safe(sum_strategies[node_id][hand])
        """

        for node in self.tree.nodes:
            node_id = node['id']
            state = self.game.node_to_state(node_id)
            if state[1] == traverser and not node['terminal']:
                start, end = self.game.get_bid_ranges(node.state)
                self.sum_strategies[node_id] *= strat_discount
                self.sum_strategies[node_id] += np.vstack([self.reach_probabilities_buffer[node_id]]*game.num_actions, 1)*self.last_strategies[node_id]
                for hand in range(self.game.num_hands):
                    for action in range(start, end):
                        self.regrets[node_id][hand][action] *= (pos_discount if self.regrets[node_id][hand][action] > 0 else neg_discount)
                    self.average_strategies[node_id][hand] = normalize_probabilities_safe(sum_strategies[node_id][hand])

        self.num_steps[traverser] += 1
    
    def multistep(self, traverser):
        for i in range(self.params.num_iters):
            self.step(i % 2)
    
    def update_value_network(self):
        self.add_training_example(0, self.get_hand_values(0))
        self.add_training_example(1, self.get_hand_values(1))
    
    def get_strategy(self):
        """
        Gets the final strategy (average strategy) that is guaranteed to converge to a Nash Equilibrium
        """
        return self.average_strategies

    def get_sampling_strategy(self):
        return self.last_strategies
    
    def get_belief_propagation_strategies(self):
        return self.last_strategies
    
    def print_strategy(self):
        return "Needs to be implemented :)"
    
    def get_hand_values(self, player_id):
        return self.root_values_means[player_id]