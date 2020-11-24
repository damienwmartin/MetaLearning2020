from game_tree import game_wrapper
from PokerRL.game.games import StandardLeduc

class Leduc(game_wrapper):

    def __init__(self, num_ranks=3, num_suits=2):
        self.num_ranks = num_ranks
        self.num_suits = num_suits
    
    def get_legal_moves(self, state):
        if state == 0:
            return 
        pass

    def get_rewards(self, state):
        pass

    def is_terminal(self, state):
        pass

    def take_action(self, state, action):
        pass

    def iter_at_node(self, node):
        pass

    def node_to_state(self, node):
        pass

    def node_to_number(self, node):
        pass