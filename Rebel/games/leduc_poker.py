from game_tree import game_wrapper
from PokerRL.game.games import StandardLeduc

class Leduc(game_wrapper):

    def __init__(self, num_ranks=3, num_suits=2):
        self.num_ranks = num_ranks
        self.num_suits = num_suits
    
    def get_legal_moves(self, game_state):
        pass

    def get_rewards():
        pass

    def is_terminal():
        pass

    def take_action():
        pass

    def iter_at_node(self, node):