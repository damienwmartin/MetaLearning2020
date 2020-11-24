from ..game_wrappers import game_wrapper
import numpy as np


class CoinGame(game_wrapper):

    def __init__(self):
        self.payoff_matrix_heads = np.array([[0.5, 0.5], [-1, 1]])
        self.payoff_matrix_tails = np.array([[-0.5, -0.5], [1, -1]])

    def get_legal_moves(self, node_name):
        if node_name == ('root', ) or node_name == ('root', 1):
            return [0, 1]
        else:
            return None
    
    def take_action(self, node_name, action):
        if action in self.get_legal_moves(node_name):
            return node_name + (action,)
        else:
            raise Exception("Action is not possible from this state")
    
    def node_to_number(self, node_name):
        if node_name == ('root', ):
            return 0
        elif node_name == ('root', 0):
            return 1
        elif node_name == ('root', 1):
            return 2
        elif node_name == ('root', 1, 0):
            return 3
        elif node_name == ('root', 1, 1):
            return 4
        else:
            raise Exception("Argument node_name is invalid")
    
    def get_rewards(self, strategy1, strategy2):
        """
        Computes the expected rewards of the players given the strategies.
        Strategies are node_id by hand
        For player 2, we include dummy "hands" that make no difference, in order for the implementation to still work
        """
        heads_payoff = np.sum(np.reshape(strategy1[:, 0], (2, 1)) * self.payoff_matrix_heads * np.reshape(np.mean(strategy2, axis=1), (1, 2)))
        tails_payoff = np.sum(np.reshape(strategy1[:, 1], (2, 1)) * self.payoff_matrix_tails * np.reshape(np.mean(strategy2, axis=1), (1, 2)))
        payoff = 0.5*heads_payoff + 0.5*tails_payoff

        return (payoff, -1*payoff)
    
    def node_to_state(self, node_name):
        """
        Converts a node to a tuple (last_action, current_player)
        """
        return (node_name[-1], int(len(node_name) > 1))
    
    def is_terminal(self, node_name):
        """
        Returns whether or not a node is a terminal node in the game
        """
        return (node_name in [('root', 0), ('root', 1, 0), ('root', 1, 1)])
    
    def sample_history(self, solver, beliefs, random_action_prob):
        """
        Randomly samples a history
        """
        tree = solver.get_tree()
        path = []

        node = tree.nodes[('root', )]
        br_sampler = np.random.randint(2)
        strategy = solver.get_sampling_strategy()

        while not node['terminal']:
            node_id = self.node_to_number(node)
            eps = np.random.uniform()
            state = self.node_to_state(node)
            if state[1] == br_sampler and eps < random_action_prob:
                action = np.random.randint(2)
            else:
                beliefs = sampling_beliefs[state[1]]
                hand = np.random.choice(beliefs.size(), 1, p=beliefs)
                policy = strategy[node_id][hand]
                action = np.random.choice(policy.size(), 1, p=policy)
                assert action in [0, 1]
            
            policy = strategy[node_id]
            sampling_beliefs[state[1]] *= policy[:, action]
            
            normalize_beliefs_inplace(sampling_beliefs[state.player_id])
            path.append((node_id, action))
            node = tree.nodes[node['id'] + (action, )]
        
        for node_id, action in path:
            policy = solver.get_belief_propagation_strategy()[node_id]
            sampling_beliefs[state[1]] = policy[:, action]
            normalize_beliefs_inplace(self.beliefs[state[1]])
    
        return path