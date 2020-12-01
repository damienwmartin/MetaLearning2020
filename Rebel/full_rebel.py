import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from games.liars_dice import LiarsDice
from games.coin_game import CoinGame
from tasks import build_value_net

EPSILON = 1e-100

class GameTree:

    def __init__(self, game, value_net, initial_beliefs):
        self.tree = nx.DiGraph()
        self.game = game
        self.tree.add_node(('root', ),
                           terminal=False,
                           subgame_terminal=False,
                           reach_prob = initial_beliefs,
                           reach_prob_buffer = np.zeros(game.num_hands),
                           cur_strategy = np.zeros((game.num_hands, game.num_actions)),
                           sum_strategy = np.zeros((game.num_hands, game.num_actions)),
                           avg_strategy = np.zeros((game.num_hands, game.num_actions)),
                           regrets = np.zeros((game.num_hands, game.num_actions)),
                           value = np.zeros(game.num_hands),
                           best_response = np.zeros((game.num_hands, game.num_actions)))
        
        self.value_net = value_net
        self.terminal_nodes = []
        self.pseudo_leaf_nodes = []

        self.cur_root_node = ('root', )
        self.cur_subgame_depth = 0

        # Neural net parameters
        self.query_size = 2 + game.num_actions + 2*game.num_hands
        self.output_size = game.num_hands
        self.net_query_buffer = []
    
    def build_depth_limited_subgame(self, root_node=('root', ), depth_limit=5):
        """
        Builds a depth-limited subgame
        """

        self.cur_root_node = root_node
        self.cur_subgame_depth = depth_limit
        self.pseudo_leaf_nodes = []
        for node_name in self.tree.nodes:
            self.tree.nodes[node_name]['subgame_terminal'] = False
        initial_depth = len(root_node)
        nodes_to_add = [root_node]
        while nodes_to_add != []:
            cur_node = nodes_to_add[0]
            
            if cur_node not in self.tree.nodes:
                self.tree.add_node(cur_node, 
                                    terminal=self.game.is_terminal(cur_node), 
                                    subgame_terminal=(self.game.is_terminal(cur_node) or len(cur_node) == initial_depth + depth_limit),
                                    reach_prob = np.zeros((2, self.game.num_hands)),
                                    reach_prob_buffer = np.zeros(self.game.num_hands),
                                    cur_strategy = np.zeros((self.game.num_hands, self.game.num_actions)),
                                    sum_strategy = np.zeros((self.game.num_hands, self.game.num_actions)),
                                    avg_strategy = np.zeros((self.game.num_hands, self.game.num_actions)),
                                    regrets = np.zeros((self.game.num_hands, self.game.num_actions)),
                                    value = np.zeros(self.game.num_hands),
                                    best_response = np.zeros((self.game.num_hands, self.game.num_actions)))
            
            if self.game.is_terminal(cur_node):
                self.terminal_nodes.append(cur_node)
            
            elif len(cur_node) == initial_depth + depth_limit:
                self.pseudo_leaf_nodes.append(cur_node)
                self.tree.nodes[cur_node]['subgame_terminal'] = True
            
            else:
                for action in self.game.get_legal_moves(cur_node):
                    nodes_to_add.append(cur_node + (action, ))
            
            nodes_to_add = nodes_to_add[1:]
        
        self.initialize_strategies()
        

    def precompute_all_reaches(self, strat='cur'):
        """
        Computes the reach probabilities within a particular subgame

        strat: can either be cur, sum, or avg
        """

        strategy = strat + '_strategy'
        nodes_to_compute = [self.cur_root_node + (action, ) for action in self.game.get_legal_moves(self.cur_root_node)]
        
        while nodes_to_compute != []:
            cur_node = nodes_to_compute[0]
            state = self.game.node_to_state(cur_node)

            prev_player = self.game.node_to_state(cur_node[:-1])[1]

            self.tree.nodes[cur_node]['reach_prob'] = 1 * self.tree.nodes[cur_node[:-1]]['reach_prob']
            if prev_player:
                self.tree.nodes[cur_node]['reach_prob'][1] *= self.tree.nodes[cur_node[:-1]][strategy][:, state[0]]
            else:
                self.tree.nodes[cur_node]['reach_prob'][0] *= self.tree.nodes[cur_node[:-1]][strategy][:, state[0]]
            
            if len(cur_node) < len(self.cur_root_node) + self.cur_subgame_depth:
                nodes_to_compute.extend([cur_node + (action, ) for action in self.game.get_legal_moves(cur_node)])
            
            nodes_to_compute = nodes_to_compute[1:]
    
    def enumerate_subgame(self):

        subgame = []
        nodes_to_add = [self.cur_root_node]

        while nodes_to_add != []:
            cur_node = nodes_to_add[0]
            subgame.append(cur_node)
            
            if not self.tree.nodes[cur_node]['terminal'] and not self.tree.nodes[cur_node]['subgame_terminal']:
                nodes_to_add.extend([cur_node + (action, ) for action in self.game.get_legal_moves(cur_node)])
            
            nodes_to_add = nodes_to_add[1:]

        return subgame
    
    def fill_reach_prob_buffer(self, traverser, strat='cur'):

        strategy = strat+'_strategy'

        self.tree.nodes[('root', )]['reach_prob_buffer'] = self.initial_beliefs[traverser]

        for node_name in self.enumerate_subgame():
            if node_name != ('root', ):
                state = self.game.node_to_state(node_name)
                prev_player = self.game.node_to_state(node_name[:-1])[1]
                if prev_player == traverser:
                    self.tree.nodes[node_name]['reach_prob_buffer'] = self.tree.nodes[node_name[:-1]]['reach_prob_buffer'] * self.tree.nodes[node_name[:-1]][strategy][:, state[0]]
                else:
                    self.tree.nodes[node_name]['reach_prob_buffer'] = self.tree.nodes[node_name[:-1]]['reach_prob_buffer']

    def query_value_net(self, traverser):
        self.net_query_buffer = []
        if self.pseudo_leaf_nodes != []:
            N = len(self.pseudo_leaf_nodes)
            scalers = []
            for node_name in self.pseudo_leaf_nodes:
                self.net_query_buffer.extend(self.write_query(node_name, traverser))
                scalers.append(np.sum(self.tree.nodes[node_name]['reach_prob'][1 - traverser]))

            scalers = torch.tensor(scalers)
            self.leaf_values = self.value_net(torch.tensor(np.reshape(np.array(self.net_query_buffer), (N, self.query_size))).float())
            self.leaf_values *= scalers.unsqueeze(1)
    
    def write_query(self, node_name, traverser):
        """
        Writes a single query to the buffer; the query corresponds to which node was seen by the traverser
        """
        state = self.game.node_to_state(node_name)
        node = self.tree.nodes[node_name]
        write_index, buffer = write_query_to(self.game, traverser, state, node['reach_prob'][0], node['reach_prob'][1])
        assert write_index == self.query_size
        return buffer
    
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

                for node_name in self.terminal_nodes:
                    beliefs = (self.tree.nodes[node_name]['reach_prob'][0] + EPSILON) / np.sum(self.tree.nodes[node_name]['reach_prob'][0] + EPSILON, axis=0, keepdims=True)
                    if node_name == ('root', 0):
                        expected_val = np.sum(beliefs*np.array([-0.5, 0.5]))
                    elif node_name == ('root', 1, 0):
                        expected_val = np.sum(beliefs*np.array([1, -1]))
                    elif node_name == ('root', 1, 1):
                        expected_val = np.sum(beliefs*np.array([-1, 1]))
                    
                    self.tree.nodes[node_name]['value'] = np.array([expected_val, expected_val])

            else:
                self.tree.nodes[('root', 0)]['value'] = np.array([0.5, -0.5])
                self.tree.nodes[('root', 1, 0)]['value'] = np.array([-1, 1])
                self.tree.nodes[('root', 1, 1)]['value'] = np.array([1, -1])

        elif isinstance(self.game, LiarsDice):
            for node_name in self.terminal_nodes:
                last_bid = self.game.node_to_state(node_name[:-1])[0]
                node = self.tree.nodes[node_name]
                node['value'] = compute_expected_terminal_values(self.game, last_bid, self.game.node_to_state(node_name)[1] != traverser, node['reach_prob'][1 - traverser])

        else:
            raise Exception("Game is currently not supported")
    
    def initialize_strategies(self):

        for node_name in self.enumerate_subgame():
            legal_moves = self.game.get_legal_moves(node_name)
            self.tree.nodes[node_name]['cur_strategy'] = np.array([[1/len(legal_moves) if i in legal_moves else 0 for i in range(self.game.num_actions)] for j in range(self.game.num_hands)])
            self.tree.nodes[node_name]['avg_strategy'] = np.array([[1/len(legal_moves) if i in legal_moves else 0 for i in range(self.game.num_actions)] for j in range(self.game.num_hands)])
        
        for traverser in [0, 1]:
            self.fill_reach_prob_buffer(traverser, strat='cur')
            
            for node_name in self.enumerate_subgame():
                node = self.tree.nodes[node_name]
                state = self.game.node_to_state(node_name)
                if not node['terminal'] and state[1] == traverser:
                    node['sum_strategy'] = node['cur_strategy']*np.reshape(node['reach_prob_buffer'], (self.game.num_hands, 1))
    
    def add_training_example(self, traverser, values, D_v):
        """
        Adds a datapoint for the value_net
        """
        value_tensor = torch.tensor(values)
        query_tensor = torch.tensor(self.write_query(self.cur_root_node, traverser))
        D_v.append((query_tensor, value_tensor))
    
    def populate_leaf_values(self):
        """
        Gets the leaf values that are not actual leaves, and reads the torch tensor from the value net result
        """
        if self.pseudo_leaf_nodes != []:
            result_acc = self.leaf_values.detach().numpy()
            for row in range(len(self.pseudo_leaf_nodes)):
                node_name = self.pseudo_leaf_nodes[row]
                self.tree.nodes[node_name]['value'] = result_acc[row]

class CFR(GameTree):
    """
    Implementation of CFR that was directly translated from the ReBeL repo

    Note: will change all instances of self.game.get_bid_ranges() to self.game.get_legal_moves() soon
    """

    def __init__(self, game, value_net, initial_beliefs, params):
        
        super().__init__(game, value_net, initial_beliefs)
        self.params = params
        self.initial_beliefs = initial_beliefs
        self.num_steps = [0, 0]

        self.root_values = [[], []]
        self.root_values_means = [[], []]
    

    def update_regrets(self, traverser):
        """
        Computes the regrets associated with a traverser for relevant nodes
        """
        self.precompute_all_reaches(strat='cur')
        self.precompute_all_leaf_values(traverser)

        for node_name in reversed(self.enumerate_subgame()):
            node = self.tree.nodes[node_name]
            if not node['terminal'] and not node['subgame_terminal']:
                state = self.game.node_to_state(node_name)
                value = np.zeros(self.game.num_hands)
                if state[1] == traverser:
                    for action in self.game.get_legal_moves(node_name):
                        child_node = self.tree.nodes[node_name + (action, )]
                        node['regrets'][:, action] += child_node['value']
                        value += child_node['value'] * node['cur_strategy'][:, action]

                    
                    for action in self.game.get_legal_moves(node_name):
                        node['regrets'][:, action] -= value
            
                else:
                    for action in self.game.get_legal_moves(node_name):
                        child_node = self.tree.nodes[node_name + (action, )]
                        value += (child_node['value'] * node['cur_strategy'][:, action])

                node['value'] = value
    
    def step(self, traverser):

        self.update_regrets(traverser)
        self.root_values[traverser] = self.tree.nodes[self.cur_root_node]['value']

        alpha = 2 / (self.num_steps[traverser] + 2) if self.params['linear_update'] else 1 / (self.num_steps[traverser] + 1)
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

        # Blackwell's Regret Matching Algorithm
        for node_name in self.enumerate_subgame():
            state = self.game.node_to_state(node_name)
            node = self.tree.nodes[node_name]
            if state[1] == traverser and not node['terminal']:
                legal_moves = self.game.get_legal_moves(node_name)
                node['cur_strategy'] = np.maximum(node['regrets'], np.array([[EPSILON if i in legal_moves else 0 for i in range(self.game.num_actions)] for j in range(self.game.num_hands)]))
                node['cur_strategy'] /= np.sum(node['cur_strategy'], axis=1, keepdims=True)
        
        self.fill_reach_prob_buffer(traverser)

        for node_name in self.enumerate_subgame():
            node = self.tree.nodes[node_name]
            state = self.game.node_to_state(node_name)
            if state[1] == traverser and not node['terminal']:

                ### The following code is only relevant if using Linear CFR or CFR-D ###
                if self.params['dcfr'] or self.params['linear_update']:
                    node['sum_strategy'] *= strat_discount
                    for hand in range(self.game.num_hands):
                        for action in range(start, end):
                            node['regrets'][hand][action] *= (pos_discount if node['regrets'][hand][action] > 0 else neg_discount)
                ### End irrelevant section ###

                node['sum_strategy'] += np.stack([node['reach_prob_buffer']]*self.game.num_actions, 1)*node['cur_strategy']
                node['avg_strategy'] = (node['sum_strategy'] + EPSILON) / np.sum(node['sum_strategy'] + EPSILON, axis=1, keepdims=True)
        
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
    
    def sample_leaf(self, node_name, random_action_prob):

        path = []
        cur_node_name = node_name
        node = self.tree.nodes[node_name]
        br_sampler = np.random.randint(2)
        beliefs = (node['reach_prob'] + EPSILON) / np.sum(node['reach_prob'] + EPSILON, axis=1, keepdims=True)
        sampling_beliefs = (node['reach_prob'] + EPSILON) / np.sum(node['reach_prob'] + EPSILON, axis=1, keepdims=True)

        while not node['terminal'] and not node['subgame_terminal']:
            eps = np.random.uniform()
            state = self.game.node_to_state(cur_node_name)
            if state[1] == br_sampler and eps < random_action_prob:
                action = np.random.choice(np.array(self.game.get_legal_moves(cur_node_name)))
            else:
                cur_beliefs = sampling_beliefs[state[1]]
                hand = np.random.choice(cur_beliefs.size, 1, p=cur_beliefs)
                policy = node['cur_strategy'][hand][0]
                action = np.random.choice(policy.size, 1, p=policy)[0]
                assert action in self.game.get_legal_moves(cur_node_name)
            
            policy = node['cur_strategy']
            sampling_beliefs[state[1]] *= np.reshape(policy[:, action], (self.game.num_hands, ))
            
            sampling_beliefs[state[1]] += EPSILON
            sampling_beliefs[state[1]] /= np.sum(sampling_beliefs[state[1]], axis=0, keepdims=True)
            path.append((node_name, action))
            cur_node_name = cur_node_name + (action, )
            node = self.tree.nodes[cur_node_name]

        for node_name, action in path:
            policy = self.tree.nodes[node_name]['cur_strategy']
            beliefs[state[1]] = np.reshape(policy[:, action], (self.game.num_hands,))
            beliefs[state[1]] += EPSILON
            beliefs[state[1]] /= np.sum(beliefs[state[1]], axis=0, keepdims=True)
    
        return cur_node_name, beliefs

    def update_value_network(self, D_v):
        self.add_training_example(0, self.get_hand_values(0), D_v)
        self.add_training_example(1, self.get_hand_values(1), D_v)
    
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
    
    def get_best_response(self, traverser):
        self.precompute_all_reaches(strat='avg')
        self.precompute_all_leaf_values(traverser)

        for node_name in reversed(list(self.tree.nodes)):
            node = self.tree.nodes[node_name]

            if not node['terminal'] and not node['subgame_terminal']:
                state = self.game.node_to_state(node_name)
                if state[1] == traverser:
                    node['best_response'] = np.zeros((self.game.num_hands, self.game.num_actions))
                    value = np.full((self.game.num_hands,), np.NINF)
                    best_action = [0 for i in range(self.game.num_hands)]
                    for action in self.game.get_legal_moves(node_name):
                        new_value = self.tree.nodes[node_name + (action, )]['value']
                        for hand in range(self.game.num_hands):
                            if new_value[hand] > value[hand]:
                                value[hand] = new_value[hand]
                                best_action[hand] = action
                    
                    node['value'] = np.array(value)
                    for hand in range(self.game.num_hands):
                        node['best_response'][hand][best_action[hand]] = 1
                
                else:
                    value = np.zeros((self.game.num_hands,))
                    beliefs = (node['reach_prob'][1 - traverser] + EPSILON) / np.sum(node['reach_prob'][1 - traverser] + EPSILON, axis=0, keepdims=True)
                    for action in self.game.get_legal_moves(node_name):
                        child_node = self.tree.nodes[node_name + (action, )]
                        value += np.sum(beliefs*node['avg_strategy'][:, action], axis=0)*np.array(child_node['value'])
                    
                    node['value'] = value
        
        return self.tree.nodes[('root',)]['value']
    
    def compute_exploitability(self):
        value0 = self.get_best_response(0)
        value1 = self.get_best_response(1)

        return 0.5*(np.mean(value0 + value1))

    

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

def normalize_probabilities_safe(x, epsilon=1e-100):
    
    total = sum(x) + len(x)*epsilon
    return [(x_i + epsilon)/total for x_i in x]

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




def rebel(game, value_net, T=1000, solver=None):

    if solver is None:
        initial_beliefs = game.get_initial_beliefs()
        params = {'dcfr': False, 'linear_update': False}
        solver = CFR(game, value_net, initial_beliefs, params)
    
    D_v = []

    cur_node = solver.tree.nodes[('root', )]
    cur_node_id = ('root', )

    while not cur_node['terminal']:

        solver.build_depth_limited_subgame(cur_node_id, depth_limit=3)
        t_sample = np.random.randint(T)

        for t in range(T):
            
            solver.step(t % 2)

            if t == t_sample:
                next_node, beliefs = solver.sample_leaf(cur_node_id, 0.1)
        
        solver.update_value_network(D_v)

        cur_node_id = next_node
        cur_node = solver.tree.nodes[cur_node_id]
    
    return D_v, solver


        

from tqdm import tqdm
def train(game, value_net, epochs, games_per_epoch, T=1000):


    solver = None
    value_optimizer = optim.Adam(value_net.parameters())
    loss_fn = nn.MSELoss()
    value_net.train()
    
    for i in range(epochs):
        print("Starting epoch ", i)
        train_x = []
        train_y = []
        for j in tqdm(range(games_per_epoch)):
			#Play a full game with rebel

            D_v, solver = rebel(game, value_net, T)
            print(len(solver.tree.nodes))
            train_x.extend([x[0] for x in D_v])
            train_y.extend([y[1] for y in D_v])
        
        train_x = torch.stack(train_x, 0).float()
        train_y = torch.stack(train_y, 0).float()
        
        value_optimizer.zero_grad()
        output = value_net.forward(train_x)
        loss = loss_fn(output, train_y)
        loss.backward()
        value_optimizer.step()
        
        if i %  4 == 1:
            print(f'Epoch {i+1}: Loss {loss}')
            PATH = f'models/liars_dice_{game.num_dice}_{game.num_faces}_{i+1}.t7'
            state = {
                'epoch': i,
                'state_dict': value_net.state_dict(),
                'optimizer': value_optimizer.state_dict(),
                'game_tree': solver.tree.nodes
            }
            torch.save(state, PATH)
    
    return solver




# Testing
if __name__ == "__main__":
    game = LiarsDice(num_dice=2, num_faces=2)

    v_net = build_value_net(game)
    end_solver = train(game, v_net, 30, 32, 250)
    print(end_solver.compute_exploitability())