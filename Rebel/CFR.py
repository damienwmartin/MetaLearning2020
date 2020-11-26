import numpy as np
import torch
from games.coin_game import CoinGame
from games.liars_dice import LiarsDice

EPSILON = 1e-100

class PartialTreeTraverser:

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

    def write_query(self, node_id, traverser):
        """
        Writes a single query to the buffer; the query corresponds to which node was seen by the traverser
        """
        state = self.game.number_to_state(node_name)
        write_index, buffer = write_query_to(self.game, traverser, state, self.reach_probabilities[0][node_id], self.reach_probabilities[1][node_id], buffer)
        assert write_index == self.query_size
        return buffer

    def precompute_reaches(self, strategy, initial_beliefs, player):
        compute_reach_probabilities(self.game, self.tree, strategy, initial_beliefs, player, self.reach_probabilities[player])
    
    def precompute_all_leaf_values(self, traverser):
        self.query_value_net(traverser)
        self.populate_leaf_values()
        self.precompute_terminal_leaves_values(traverser)

    def precompute_terminal_leaves_values(self, traverser):
        """
        Computes the value of each terminal node
        """

        if isinstance(self.game, CoinGame):
            if traverser:
                # Beliefs of H vs T for the person selling

                for node_id in [1, 3, 4]:
                    beliefs = normalize_probabilities_safe(self.reach_probabilities[0][node_id]) # Array of [P(heads), P(tails)]?

                    # Calculate EV

                    # Use beliefs for traverser 2
            else:
                self.traverser_values[1] = np.array([0.5, -0.5])
                self.traverser_values[3] = np.array([-1, 1])
                self.traverser_values[4] = np.array([1, -1])

        if isinstance(self.game, LiarsDice):
            for node_name in self.terminal_indices:
                last_bid = self.game.node_to_state(node_name[:-1])[0]
                node_id = self.game.node_to_number(node_name)
                self.traverser_values[node_id] = compute_expected_terminal_values(self.game, last_bid, self.game.node_to_state(node_name)[1] != traverser, self.reach_probabilities[1 - traverser][node_id])

    def query_value_net(self, traverser):
        if self.pseudo_leaves_indices != []:
            N = len(self.pseudo_leaves_indices)
            scalers = []
            for row in range(N):
                node_id = self.pseudo_leaves_indices(row)
                self.net_query_buffer.extend(self.write_query(node_id, traverser))
                scalers.append(np.sum(self.reach_probabilities[1 - traverser][node_id]))
            scalers = torch.tensor(scalers)
            self.leaf_values = self.value_net.compute_values(np.reshape(np.array(self.net_query_buffer), (N, self.query_size)))
            self.leaf_values *= scalers.unsqueeze(1)
    

    def populate_leaf_values(self):
        if self.pseudo_leaves_indices != []:
            result_acc = self.leaf_values.cpu().numpy()
            for row in range(len(self.pseudo_leaves_indices)):
                node_name = self.pseudo_leaves_indices[row]
                node_id = self.game.node_to_number(node_name)
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
        
        self.average_strategies = get_uniform_strategy(game, self.tree)
        self.last_strategies = self.average_strategies
        self.sum_strategies = get_uniform_reach_weighted_strategy(game, self.tree, self.initial_beliefs)
        self.regrets = np.zeros((len(tree), game.num_hands, game.num_actions))
        self.reach_probabilities_buffer = np.zeros((len(tree), game.num_hands))

        self.root_values = [[], []]
        self.root_values_means = [[], []]
    
    def update_regrets(self, traverser):
        """
        Computes the regrets associated with a traverser and stores the result in self.regrets
        """
        self.precompute_reaches(self.last_strategies, self.initial_beliefs, traverser)
        self.precompute_all_leaf_values(traverser)

        for public_node_name in self.tree.nodes:
            public_node = self.tree.nodes[public_node_name]
            public_node_id = self.game.node_to_number(public_node_name)
            if not public_node['subgame_terminal'] and not public_node['terminal']:
                state = self.game.node_to_state(public_node_name)
                start, end = self.game.get_bid_ranges(public_node_name)
                value = np.zeros_like(self.traverser_values[public_node_id]) # array of size (num_hands, )
                action_values = np.transpose(np.array([self.traverser_values[child_node_id] if child_node_id else [0]*game.num_hands for action, child_node_id in self.game.iter_at_node(public_node_id)])) # array of size (num_hands, num_actions)
                if state[1] == traverser:
                    self.regrets[public_node_id] += action_values
                    # (num_hands, num_actions) -> (num_hands, num_actions)
                    value += np.sum(action_values * self.last_strategies[public_node_id], axis=1)
                    #(num_hands, )  [SUM 1](num_hands, num_actions)*(num_hands, num_actions)
                    self.regrets[public_node_id] -= np.stack([value if i >= start and i < end else 0 for i in range(self.game.num_actions)], 1) #(num_hands, num_actions)
                    
                else:
                    assert state[1] == 1 - traverser
                    value += np.sum(action_values, axis=1)  # (num_hands) + [SUM 1] (num_hands, num_actions)
            
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
        alpha = 2 /(self.num_steps[traverser] + 2) if self.params['linear_update'] else 1 / (self.num_steps[traverser] + 1)
        self.root_values_means[traverser] = resize(self.root_values_means[traverser], len(self.root_values[traverser]))

        self.root_values_means[traverser] += alpha*(self.root_values[traverser] - self.root_values_means[traverser])

        """
        for i in range(len(self.root_values[traverser])):
            self.root_values_means[traverser][i] += (self.root_values[traverser][i] - self.root_values_means[traverser][i])*alpha
        """

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
        
        """
        for node_id in range(len(self.tree)):
            if self.tree[node_id].num_children() and self.tree[node_id].state.player_id == traverser:
                start, end = self.game.get_bid_ranges(self.tree[node_id].state)
                for hand in range(self.game.num_hands())
                    for action in range(start, end):
                        self.last_strategies[node_id][hand][action] = max(regrets[node_id][hand][action], EPSILON)
                    last_strategies[node][hand] = normalize_probabilities_safe(last_strategies[node][hand])
        """

        for node_name in self.tree.nodes:
            state = self.game.node_to_state(node_name)
            node = self.tree.nodes[node_name]
            if state[1] == traverser and not node['terminal']:
                start, end = self.game.get_bid_ranges(node_name)
                node_id = self.game.node_to_number(node_name)
                self.last_strategies[node_id] = np.maximum(self.regrets[node_id], np.array([EPSILON if i >= start and i < end else 0 for i in range(self.game.num_actions)]))
                for hand in range(self.game.num_hands):
                    self.last_strategies[node_id][hand] = normalize_probabilities_safe(self.last_strategies[node_id][hand])

        compute_reach_probabilities(self.game, self.tree, self.last_strategies, self.initial_beliefs[traverser], traverser, self.reach_probabilities_buffer)

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

        for node_name in self.tree.nodes:
            node = self.tree.nodes[node_name]
            state = self.game.node_to_state(node_name)
            if state[1] == traverser and not node['terminal']:
                node_id = self.game.node_to_number(node_name)
                start, end = self.game.get_bid_ranges(node_name)
                self.sum_strategies[node_id] *= strat_discount
                self.sum_strategies[node_id] += np.vstack([self.reach_probabilities_buffer[node_id]]*self.game.num_actions)*self.last_strategies[node_id]
                for hand in range(self.game.num_hands):
                    if self.params['dcfr'] or self.params['linear_update']:
                        for action in range(start, end):
                            self.regrets[node_id][hand][action] *= (pos_discount if self.regrets[node_id][hand][action] > 0 else neg_discount)
                    self.average_strategies[node_id][hand] = normalize_probabilities_safe(self.sum_strategies[node_id][hand])

        self.num_steps[traverser] += 1
    
    def multistep(self):
        for i in range(self.params['num_iters']):
            self.step(i % 2)
            if i % 10 == 0:
                print('Iteration %d', i)
                print("Player 1 strategy:", self.get_strategy()[0])
                print("Player 2 strategy:", self.get_strategy()[2])
    
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
 
    for node_name in tree.nodes:
        node_id = game.node_to_number(node_name)
        if node_name != ('root', ):
            state = game.node_to_state(node_name)
            last_action_player_id = game.node_to_state(node_name[:-1])[1]
            parent_node_id = game.node_to_number(node_name[:-1])


            if player == last_action_player_id:
                reach_probabilities[node_id] = reach_probabilities[parent_node_id]*strategy[parent_node_id, :, state[0]]
            
            else:
                reach_probabilities[node_id] = reach_probabilities[parent_node_id]
        else:
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
            if not node['terminal'] and state[1] == traverser:
                for action in game.get_legal_moves(node_name):
                    strategy[node_id, :, action] *= reach_probabilities_buffer[node_id]

        """
        for node in range(len(tree)):
            if tree[node].num_children() and tree_node.state.player_id == traverser:
                action_begin, action_end = game.get_bid_ranges(tree[node].state)
                for hand in range(game.num_hands()):
                    for action in range(action_begin, action_end):
                        strategy[node][hand][action] *= reach_probabilities_buffer[node][hand]
        """
    
    return strategy


def compute_expected_terminal_values(game, last_bid, inverse, op_reach_probabilities):
    """
    Computes the expected terminal values for each node, for each hand

    op_reach_probabilities -> input from precompute_terminal_leaf values was probability of reaching the node for each hand
    """
    inv = 2*int(inverse) - 1
    values = self.game.compute_win_probability(last_bid, op_reach_probabilities)
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