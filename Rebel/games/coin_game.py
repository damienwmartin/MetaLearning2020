from game_wrappers import game_wrapper
import numpy as np


class CoinGame(game_wrapper):
    #PBS is the last move, -1->start, 0->play, 1->sell, 2->guess heads, 3 -> guess tails

    def __init__(self):
        self.payoff_matrix_heads = np.array([[0.5, 0.5], [-1, 1]])
        self.payoff_matrix_tails = np.array([[-0.5, -0.5], [1, -1]])

        self.num_hands = 2
        self.num_actions = 2

    """
    def get_legal_moves(self, pbs):
        if pbs.public in [-1, 0]:
            return([0, 1])
        else:
            return([])
    """

    def get_legal_moves(self, node_name):
        if node_name == ('root', ) or node_name == ('root', 1):
            return [0, 1]
        else:
            return []
    
    def get_bid_ranges(self, node_name):
        if node_name == ('root', ) or node_name == ('root', 1):
            return (0, 2)
        else:
            return (2, 2)
    
    def take_action(self, pbs, action):
        if action in self.get_legal_moves(pbs):
            if pbs.public == -1:
                return(action)
            if pbs.public == 0:
                return(2+action)
        else:
            raise Exception("Action is not possible from this state")
    
    #def get_rewards(self, strategy1, strategy2):
    #    """
    #    Computes the expected rewards of the players given the strategies.
    #    Strategies are node_id by hand
    #    For player 2, we include dummy "hands" that make no difference, in order for the implementation to still work
    #    """
    #    heads_payoff = np.sum(np.reshape(strategy1[:, 0], (2, 1)) * self.payoff_matrix_heads * np.reshape(np.mean(strategy2, axis=1), (1, 2)))
    #    tails_payoff = np.sum(np.reshape(strategy1[:, 1], (2, 1)) * self.payoff_matrix_tails * np.reshape(np.mean(strategy2, axis=1), (1, 2)))
    #    payoff = 0.5*heads_payoff + 0.5*tails_payoff
    #
    #    return (payoff, -1*payoff)
    
    def get_rewards(self, pbs):
        '''
        First number for outcome if heads second number for tails
        rewards given with respect to player 1
        '''
        if pbs.public == 1:
            return(np.array([.5,-.5]))
        if pbs.public == 2:
            return(np.array([-1,1]))
        if pbs.public == 3:
            return(np.array([1,-1]))



    def node_to_state(self, node_name):
        """
        Converts a node to a tuple (last_action, current_player)
        """
        return (node_name[-1], (len(node_name) - 1) % 2)
    
    """
    def is_terminal(self, pbs):
        Returns whether or not a node is a terminal node in the game
        return(pbs.public in [1,2,3])
    """

    def is_terminal(self, node_name):
        return node_name in [('root', 0), ('root', 1, 0), ('root', 1, 1)]
    
    def iter_at_node(self, node_id):
        """
        Iterates through all possible action, child_node_id pairs for a particular node. Actions that are not possible at a node are still iterated through, but the child_node_id is None
        """
        if node_id == 0:
            yield (0, 1)
            yield (1, 2)
        elif node_id == 2:
            yield (0, 3)
            yield (1, 4)
        else:
            yield (0, None)
            yield (1, None)
   
    def compute_win_probability(self, node_name, beliefs):
        """
        Given the probability that nodes are reached, computes the probability that the traverser wins
        """

        if node_name == ('root', 0):
            return 0
        elif node_name == ('root', 1, 0):
            return 0
        elif node_name == ('root', 1, 1):
            return 0
        else:
            raise Exception("The node entered is not a terminal node of the Coin Game.")
    
    def get_initial_beliefs(self):
        return np.array([[0.5, 0.5], [0.5, 0.5]])

        