import numpy as np
import torch
from games.coin_game import CoinGame
from games.liars_dice import LiarsDice

EPSILON = 1e-100

class PartialTreeTraverser:

    """
    This class is a base class that contains methods used for game tree traversal and associated calculations.
    """

    def __init__(self, game, tree, value_net):

        self.game = game
        self.tree = tree.tree
        self.query_size = get_query_size(game)
        self.output_size = game.num_hands

        self.pseudo_leaves_indices = []
        self.terminal_indices = []
        self.value_net = value_net
        self.net_query_buffer = []

        if value_net:
            for node_name in self.tree.nodes:
                node = self.tree.nodes[node_name]
                if node['subgame_terminal'] and not node['terminal']:
                    # Pseudo-leaves are nodes which are leaves in our depth-limited subgame but not actually terminal states in the game
                    self.pseudo_leaves_indices.append(node_name)
        else:
            for node_id in self.tree.nodes:
                node = self.tree.nodes[node_id]
                if node['subgame_terminal'] and not node['terminal']:
                    raise Exception("Found a node that is a non-final leaf. Either provide a value net or increase max_depth.")

        for node_name in self.tree.nodes:
            node = self.tree.nodes[node_name]
            if node['terminal']:
                self.terminal_indices.append(node_name)
        
        self.leaf_values = torch.zeros((len(self.pseudo_leaves_indices), self.output_size))
        self.traverser_values = np.zeros((len(tree), game.num_hands))
        self.reach_probabilities = (np.zeros((len(tree), game.num_hands)), np.zeros((len(tree), game.num_hands)))
    
    def add_training_example(self, traverser, values):
        """
        Adds a datapoint for the value_net
        """
        value_tensor = torch.tensor(values)
        query_tensor = torch.tensor(self.write_query(('root', ), traverser))
        self.value_net.add_training_example(query_tensor, value_tensor)

    def write_query(self, node_name, traverser):
        """
        Writes a single query to the buffer; the query corresponds to which node was seen by the traverser
        """
        state = self.game.node_to_state(node_name)
        node_id = self.game.node_to_number(node_name)
        write_index, buffer = write_query_to(self.game, traverser, state, self.reach_probabilities[0][node_id], self.reach_probabilities[1][node_id])
        assert write_index == self.query_size
        return buffer

    def precompute_reaches(self, strategy, initial_beliefs, player):
        compute_reach_probabilities(self.game, self.tree, strategy, initial_beliefs, player, self.reach_probabilities[player])
    
    def precompute_all_reaches(self, strategy, initial_beliefs):
        self.precompute_reaches(strategy, initial_beliefs[0], 0)
        self.precompute_reaches(strategy, initial_beliefs[1], 1)
    
    def precompute_all_leaf_values(self, traverser):
        self.query_value_net(traverser)
        self.populate_leaf_values()
        self.precompute_terminal_leaves_values(traverser)

    def precompute_terminal_leaves_values(self, traverser):
        """
        Computes the expected value of each terminal node, according to the traverser
        """

        if isinstance(self.game, CoinGame):
            if traverser:
                # Beliefs of H vs T for the person selling

                for node_id in [1, 3, 4]:
                    beliefs = normalize_probabilities_safe(self.reach_probabilities[0][node_id]) # Array of [P(heads), P(tails)]?

                    if node_id == 1:
                        expected_val = np.sum(beliefs*np.array([-0.5, 0.5]))
                    elif node_id == 3:
                        expected_val = np.sum(beliefs*np.array([1, -1]))
                    elif node_id == 4:
                        expected_val = np.sum(beliefs*np.array([-1, 1]))
                    
                    self.traverser_values[node_id] = np.array([expected_val, expected_val])

            else:
                self.traverser_values[1] = np.array([0.5, -0.5])
                self.traverser_values[3] = np.array([-1, 1])
                self.traverser_values[4] = np.array([1, -1])

        elif isinstance(self.game, LiarsDice):
            for node_name in self.terminal_indices:
                last_bid = self.game.node_to_state(node_name[:-1])[0]
                node_id = self.game.node_to_number(node_name)
                self.traverser_values[node_id] = compute_expected_terminal_values(self.game, last_bid, self.game.node_to_state(node_name)[1] != traverser, self.reach_probabilities[1 - traverser][node_id])

        else:
            raise Exception("Game is currently not supported")

    def query_value_net(self, traverser):
        self.net_query_buffer = []
        if self.pseudo_leaves_indices != []:
            N = len(self.pseudo_leaves_indices)
            scalers = []
            for node_name in self.pseudo_leaves_indices:
                self.net_query_buffer.extend(self.write_query(node_name, traverser))
                node_id = self.game.node_to_number(node_name)
                scalers.append(np.sum(self.reach_probabilities[1 - traverser][node_id]))
            scalers = torch.tensor(scalers)
            self.leaf_values = self.value_net(torch.tensor(np.reshape(np.array(self.net_query_buffer), (N, self.query_size))).float())
            self.leaf_values *= scalers.unsqueeze(1)
    

    def populate_leaf_values(self):
        """
        Gets the leaf values that are not actual leaves, and reads the torch tensor from the value net result
        """
        if self.pseudo_leaves_indices != []:
            result_acc = self.leaf_values.detach().numpy()
            for row in range(len(self.pseudo_leaves_indices)):
                node_name = self.pseudo_leaves_indices[row]
                node_id = self.game.node_to_number(node_name)
                self.traverser_values[node_id] = result_acc[row]

class CFR(PartialTreeTraverser):
    """
    Implementation of CFR that was directly translated from the ReBeL repo

    Note: will change all instances of self.game.get_bid_ranges() to self.game.get_legal_moves() soon
    """

    def __init__(self, game, tree, value_net, beliefs, params):
        super().__init__(game, tree, value_net)
        self.params = params
        self.initial_beliefs = beliefs
        self.num_steps = [0, 0]
        
        self.average_strategies = get_uniform_strategy(game, self.tree)
        self.last_strategies = get_uniform_strategy(game, self.tree)
        self.sum_strategies = get_uniform_reach_weighted_strategy(game, self.tree, self.initial_beliefs)
        self.regrets = np.zeros((len(tree), game.num_hands, game.num_actions))
        self.reach_probabilities_buffer = np.zeros((len(tree), game.num_hands))

        self.root_values = [[], []]
        self.root_values_means = [[], []]
    
    def update_regrets(self, traverser, i):
        """
        Computes the regrets associated with a traverser and stores the result in self.regrets
        """
        self.precompute_all_reaches(self.last_strategies, self.initial_beliefs) # Computes the probability that a certain node is reached
        self.precompute_all_leaf_values(traverser) # Computes the expected value for all leaf nodes

        for public_node_name in reversed(list(self.tree.nodes)):
            public_node = self.tree.nodes[public_node_name]
            public_node_id = self.game.node_to_number(public_node_name)
            if not public_node['subgame_terminal'] and not public_node['terminal']:
                state = self.game.node_to_state(public_node_name)
                start, end = self.game.get_bid_ranges(public_node_name)
                value = np.zeros_like(self.traverser_values[public_node_id]) # Will store the value of the strategy for each hand
                if state[1] == traverser:
                    for action in range(start, end):
                        child_node_id = self.game.node_to_number(public_node_name + (action, ))
                        self.regrets[public_node_id, :, action] += self.traverser_values[child_node_id] # Utility of the action
                        value += self.traverser_values[child_node_id] * self.last_strategies[public_node_id, :, action] # Adds the contribution of the action to the current value. This value is basically  SUM(P(a)*v(a)) for actions a, where P(a) is the probability and v(a) is the value.
                        
                    for action in range(start, end):
                        self.regrets[public_node_id, :, action] -= value # subtract the current strategy value from the regrets
                else:
                    # In this case, the traverser is not the player at this node, so there are no regrets to worry about
                    for action in range(start, end):
                        child_node_id = self.game.node_to_number(public_node_name + (action, ))
                        value += self.traverser_values[child_node_id]*self.last_strategies[public_node_id, :, action] # In the original code, there was no multiplication by self.last_strategies, but including it made this work for the Coin Game
            
                self.traverser_values[public_node_id] = value
    
    def step(self, traverser, i):
        """
        Does a step of the CFR algorithm, and updates the average policies
        """
        self.update_regrets(traverser, i) # Computes the regrets associated with the current strategy
        self.root_values[traverser] = self.traverser_values[0] # Sets the ev of the root node (and thus of the strategy)

        # Updates the average using a factor of alpha, which depends on if we use LCFR or normal CFR
        alpha = 2 /(self.num_steps[traverser] + 2) if self.params['linear_update'] else 1 / (self.num_steps[traverser] + 1)
        
        self.root_values_means[traverser] = resize(self.root_values_means[traverser], len(self.root_values[traverser]))
        self.root_values_means[traverser] += alpha*(self.root_values[traverser] - self.root_values_means[traverser])


        ### This section is relevant only if you use Linear CFR or CFR-D ###
        pos_discount = 1
        neg_discount = 1
        strat_discount = 1

        num_strategies = self.num_steps[traverser] + 1
        if self.params['linear_update']:
            pos_discount = num_strategies / (num_strategies + 1)
            neg_discount = pos_discount
            strat_discount = pos_discount
        elif self.params['dcfr']:
            if self.params['dcfr_alpha'] < 5:
                pos_discount = num_strategies**self.params['dcfr_alpha'] / (num_strategies**self.params['dcfr_alpha'] + 1)
            if self.params['dcfr_beta'] > -5:
                neg_discount = num_strategies**self.params['dcfr_beta'] / (num_strategies**self.params['dcfr_beta'] + 1)
            strat_discount = (num_strategies / (num_strategies + 1))**self.params['dcfr_gamma']
        ### End irrelevant section ###
    
        # Blackwell's Regret-Matching Algorithm
        for node_name in self.tree.nodes:
            state = self.game.node_to_state(node_name)
            node = self.tree.nodes[node_name]
            if state[1] == traverser and not node['terminal']:
                start, end = self.game.get_bid_ranges(node_name)
                node_id = self.game.node_to_number(node_name)
                self.last_strategies[node_id] = np.maximum(self.regrets[node_id], np.array([EPSILON if i >= start and i < end else 0 for i in range(self.game.num_actions)])) # Get only the positive portions of the regrets
                self.last_strategies[node_id] /= np.sum(self.last_strategies[node_id], axis=1, keepdims=True) # Get the probability distribution across all actions, proportional to regrets
        
        compute_reach_probabilities(self.game, self.tree, self.last_strategies, self.initial_beliefs[traverser], traverser, self.reach_probabilities_buffer)

        # Updates average strategy
        for node_name in self.tree.nodes:
            node = self.tree.nodes[node_name]
            state = self.game.node_to_state(node_name)
            if state[1] == traverser and not node['terminal']:
                node_id = self.game.node_to_number(node_name)
                start, end = self.game.get_bid_ranges(node_name)
                
                ### The following code is only relevant if using Linear CFR or CFR-D ###
                if self.params['dcfr'] or self.params['linear_update']:
                    self.sum_strategies[node_id] *= strat_discount
                    for hand in range(self.game.num_hands):
                        for action in range(start, end):
                            self.regrets[node_id][hand][action] *= (pos_discount if self.regrets[node_id][hand][action] > 0 else neg_discount)
                ### End irrelevant section ###

                self.sum_strategies[node_id] += np.stack([self.reach_probabilities_buffer[node_id]]*self.game.num_actions, 1)*self.last_strategies[node_id]
                self.average_strategies[node_id] = self.sum_strategies[node_id] / np.sum(self.sum_strategies[node_id], axis=1, keepdims=True)
        
        self.num_steps[traverser] += 1
    
    def multistep(self):
        """
        Does multiple steps of the CFR algorithm. The exact number is specified in the params dictionary
        """
        for i in range(self.params['num_iters']):
            self.step(i % 2, i)
            if i % 100 == 0:
                print('Iteration %d', i)
                print("Player 1 average strategy", self.get_strategy()[0])
                print("Player 2 average strategy", self.get_strategy()[2])


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
    
    def get_belief_propagation_strategy(self):
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
 
    for node_name in tree.nodes:
        node_id = game.node_to_number(node_name)
        if node_name != ('root', ):
            state = game.node_to_state(node_name)
            last_action_player_id = game.node_to_state(node_name[:-1])[1]
            parent_node_id = game.node_to_number(node_name[:-1])

            if player == last_action_player_id:
                # The probability of reaching this node is the probability of reaching the parent node * the probability that the correct action is taken at this node
                reach_probabilities[node_id] = reach_probabilities[parent_node_id]*strategy[parent_node_id, :, state[0]]
            
            else:
                # We only care about contributions to the reach probability from the current player
                reach_probabilities[node_id] = reach_probabilities[parent_node_id]
        else:
            # In this case, we are dealing with the root node, which makes this our initial_beliefs
            reach_probabilities[node_id] = initial_beliefs


def get_uniform_reach_weighted_strategy(game, tree, initial_beliefs):
    """
    Gets a strategy that is weighted based on reach probabilities
    """

    strategy = get_uniform_strategy(game, tree)
    reach_probabilities_buffer = np.zeros((len(tree.nodes), game.num_hands))
    
    for traverser in [0, 1]:
        compute_reach_probabilities(game, tree, strategy, initial_beliefs[traverser], traverser, reach_probabilities_buffer)

        for node_name in tree.nodes:
            node = tree.nodes[node_name]
            state = game.node_to_state(node_name)
            node_id = game.node_to_number(node_name)
            start, end = game.get_bid_ranges(node_name)
            if not node['terminal'] and state[1] == traverser:
                # for action in game.get_legal_moves(node_name):
                for action in range(start, end):
                    strategy[node_id, :, action] *= reach_probabilities_buffer[node_id]
    
    return strategy


def compute_expected_terminal_values(game, last_bid, inverse, op_reach_probabilities):
    """
    Computes the expected terminal values for each node, for each hand

    op_reach_probabilities -> input from precompute_terminal_leaf values was probability of reaching the node for each hand
    """
    inv = 2*int(inverse) - 1
    values = compute_win_probability(game, last_bid, op_reach_probabilities)
    belief_sum = sum(op_reach_probabilities)

    # Normalize values based on the sum of op_reach_probabilities
    for i in range(len(values)):
        values[i] = (2*values[i] - belief_sum)*inv
        
    return values


def get_query_size(game):

    return 1 + 1 + game.num_actions + game.num_hands*2

def get_uniform_strategy(game, tree):
    """
    Returns a strategy that picks actions uniformly at random at each node from all possible legal actions
    """

    strategy = np.zeros((len(tree), game.num_hands, game.num_actions))
    for node_name in tree.nodes:
        node_id = game.node_to_number(node_name)
        start, end = game.get_bid_ranges(node_name)
        strategy[node_id] = np.array([[1/(end - start) if j >= start and j < end else 0 for j in range(game.num_actions)] for i in range(game.num_hands)])
    
    return strategy


def resize(x, size):
    """
    Python implementation of the C++ standard .resize() operation
    """
    N = len(x)
    if size > N:
        y = x[:]
        y.extend([0]*(size - N))
        return y
    else:
        return x[:size]


def normalize_probabilities_safe(x, epsilon=1e-200):
    
    total = sum(x) + len(x)*epsilon
    return [(x_i + epsilon)/total for x_i in x]


def compute_win_probability(game, action, beliefs):
    unpacked_action = game.unpack_action(action)
    believed_counts = [0 for i in range(game.total_num_dice + 1)]
    for hand in range(len(beliefs)):
        matches = game.num_matches(hand, unpacked_action[1])
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
