from ..game_wrappers import game_wrapper
import numpy as np


class CoinGame(game_wrapper):

    def __init__(self):
        super().__init__()
        self.payoff_matrix_heads = np.array([[0.5, 0.5], [-1, 1]])
        self.payoff_matrix_tails = np.array([[-0.5, -0.5], [1, -1]])

    def get_legal_moves(self, node_name):
        if node_name == ('root', )
            return ['sell', 'bet']
        elif node_name == ('root', 'bet')
            return ['heads', 'tails']
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
        elif node_name == ('root', 'sell'):
            return 1
        elif node_name == ('root', 'bet'):
            return 2
        elif node_name == ('root', 'bet', 'heads'):
            return 3
        elif node_name == ('root', 'bet', 'tails'):
            return 4
        else:
            raise Exception("Argument node_name is invalid")
    
    def get_rewards(self, strategy1, strategy2):
        """
        Computes the expected rewards of the players given the strategies.
        Strategies for the first person depend on 
        """
        heads_payoff = np.sum(np.reshape(strategy1[0], (2, 1)) * self.payoff_matrix_heads * np.reshape(strategy2, (1, 2)))
        tails_payoff = np.sum(np.reshape(strategy1[1], (2, 1)) * self.payoff_matrix_tails * np.reshape(strategy2, (1, 2)))
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
        return (node_name in [('root', 'sell'), ('root', 'bet', 'heads'), ('root', 'bet', 'tails')])
    
    def sample_history(self):
