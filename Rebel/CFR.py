import numpy as np
import torch
from .game_tree import node_to_number

EPSILON = 1e-100

class PartialTreeTraverser:

    def __init__(self, game, tree, value_net):

        self.game = game
        self.tree = tree
        self.query_size = get_query_size(game)
        self.output_size = game.num_hands

        self.pseudo_leaves_indices = []
        self.terminal_indices = []
        self.value_net = value_net
        self.net_query_buffer = []

        if value_net:
            for node in self.tree.nodes:
                node_id = node['id']
                if node['subgame_terminal'] and not node['terminal']:
                    # Pseudo-leaves are nodes which are leaves in our depth-limited subgame but not actually terminal states in the game
                    self.pseudo_leaves_indices.append(node_id)
        else:
            for node in self.tree.nodes:
                if node['subgame_terminal'] and not node['terminal']:
                    raise Exception("Found a node that is a non-final leaf. Either provide a value net or increase max_depth.")

        for node in self.tree.nodes:
            node_id = node['id']
            if node['terminal']:
                self.terminal_indices.append(node_id)
        
        self.leaf_values = torch.zeros(len(pseudo_leaves_indices), output_size)
        self.traverser_values = np.zeros(len(tree), game.num_hands)
        self.reach_probabilities = (np.zeros(len(tree), game.num_hands), np.zeros(len(tree), game.num_hands))
    
    def add_training_example(self, traverser, values):
        """
        Adds a datapoint for the value_net
        """
        value_tensor = torch.tensor(values)
        query_tensor = torch.tensor(self.write_query(('root', ), traverser))
        self.value_net.add_training_example(query_tensor, value_tensor)

    def write_query(self, node_id, traverser):
        """
        Writes a single query to the buffer; the query corresponds to which node was seen by the traverser
        """
        state = self.game.node_to_state(node_name)
        node_id = node_to_number(node_name)
        write_index, buffer = write_query_to(self.game, traverser, state, self.reach_probabilities[0][node_id], self.reach_probabilities[1][node_id], buffer)
        assert write_index == self.query_size
        return buffer

    def precompute_reaches(self, strategy, initial_beliefs, player):
        compute_reach_probabilities(tree, strategy, initial_beliefs, player, reach_probabilities[player])
    

    def precompute_all_leaf_values(self, traverser):
        self.query_value_net(traverser)
        self.populate_leaf_values()
        self.precompute_terminal_leaves_values()

    def precompute_terminal_leaves_values(traverser):

        for node_id in self.terminal_indices:
            last_bid = self.tree[self.tree[node_id].parent].state.last_bid
            traverser_values[node_id] = compute_expected_terminal_values(game, last_bid, tree[node_id].state.player_id != traverser, reach_probabilities[1 - traverser][node_id])
    
    def query_value_net(self, traverser):
        if self.pseudo_leaves_indices != []:
            N = len(self.pseudo_leaves_indices)
            scalers = []
            for row in range(N):
                node_id = self.pseudo_leaves_indices(row):
                self.net_query_buffer.extend(self.write_query(node_id, traverser))
                scalers.append(np.sum(self.reach_probabilities[1 - traverser][node_id]))
            scalers = torch.tensor(scalers)
            self.leaf_values = self.value_net.compute_values(np.reshape(np.array(self.net_query_buffer), (N, self.query_size)))
            self.leaf_values *= scalers.unsqueeze(1)
    

    def populate_leaf_values(self):
        if self.pseudo_leaves_indices != []:
            result_acc = self.leaf_values.cpu().numpy()
            for row in range(len(self.pseudo_leaves_indices)):
                node_id = pseudo_leaves_indices[row]
                self.traverser_values[node_id] = result_acc[row]


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
            if not public_node['subgame_terminal'] and not public_node['terminal']:
                state = self.game.node_to_state(public_node)
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
                start, end = self.game.get_bid_ranges(state)
                self.sum_strategies[node_id] *= strat_discount
                self.sum_strategies[node_id] += np.vstack([self.reach_probabilities_buffer[node_id]]*game.num_actions, 1)*self.last_strategies[node_id]
                for hand in range(self.game.num_hands):
                    if params.dcfr:
                        for action in range(start, end):
                            self.regrets[node_id][hand][action] *= (pos_discount if self.regrets[node_id][hand][action] > 0 else neg_discount)
                    self.average_strategies[node_id][hand] = normalize_probabilities_safe(self.sum_strategies[node_id][hand])

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
    
    def get_tree(self):
        return self.tree



def write_query_to(game, traverser, state, reaches1, reaches2):
    """
    Creates the query tensor to be added to the value net
    """

    buffer = [state[1], traverser]
    for action in range(game.num_actions):
        buffer.append(int(action == state[0]))

    buffer.extend(normalize_probabilities_safe(reaches1, EPSILON))
    buffer.extend(normalize_probabilities_safe(reaches2, EPSILON))

    write_index = len(buffer)

    return write_index, buffer


def compute_reach_probabilities(game, tree, strategy, initial_beliefs, player, reach_probabilities):
    """
    Recomputes the probability that a certain node is reached
    """
 
    for node in tree.nodes:
        if node['id'] != ('root', ):
            parent_node = tree.nodes[node['id'][:-1]]
            state = game.node_to_state(node)
            last_action_player_id = game.node_to_state(parent_node)[1]
            last_bid = state[0]

            node_id = game.node_to_number(node)
            parent_node_id = game.node_to_number(parent_node)


            if player == last_action_player_id:
                reach_probabilities[node_id] = reach_probabilities[parent_node_id]*strategy[parent_node_id, :, last_action]
            
            else:
                reach_probabilities[node_id] = reach_probabilities[parent_node_id]
        else:
            reach_probabilities[node_id] = initial_beliefs


def get_uniform_reach_weighted_strategy(game, tree, initial_beliefs):
    """
    Gets a strategy that is weighted based on reach probabilities
    """

    strategy = get_uniform_strategy(game, tree)
    reach_probabilities_buffer = np.zeros(len(tree), game.num_hands)
    
    for traverser in [0, 1]:
        compute_reach_probabilities(tree, strategy, initial_beliefs[traverser], traverser, reach_probabilities_buffer)

        for node in tree.nodes:
            state = game.node_to_state(node)
            if not node['terminal'] and state[1] == traverser:
                for action in game.get_legal_moves(node):
                    strategy[node, :, action] *= reach_probabilities_buffer[node]
        
        return strategy

        """
        for node in range(len(tree)):
            if tree[node].num_children() and tree_node.state.player_id == traverser:
                action_begin, action_end = game.get_bid_range(tree[node].state)
                for hand in range(game.num_hands()):
                    for action in range(action_begin, action_end):
                        strategy[node][hand][action] *= reach_probabilities_buffer[node][hand]
        """
    
    return strategy


def compute_expected_terminal_values(game, last_bid, inverse, op_reach_probabilities):
    
    inv = 2*int(inverse) - 1
    values = compute_win_probability(game, last_bid, op_reach_probabilities)
    belief_sum = sum(op_reach_probabilities)

    for i in range(len(values)):
        values[i] = (2*values[i] - belief_sum)*inv
    
    return values


def get_query_size(game):

    return 1 + 1 + game.num_actions() + game.num_hands()*2