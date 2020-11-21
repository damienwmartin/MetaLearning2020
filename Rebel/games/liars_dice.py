from game_wrappers import game_wrapper
# import LiarsDice_C from Modified C Code?

class LiarsDice(game_wrapper):
    """
    Simple Wrapper for Liar's Dice that is an extension of our generic game_wrapper class
    PBS are given as (action_id, player_id) pairs
    game_states are given as (action_id, player_id, hand_1, hand_2)
    """

    def __init__(self, num_dice, num_faces):
        super().__init__()
        self.game = LiarsDice_C(num_dice, num_faces)
    
    def get_legal_moves(self, pbs):
        return self.game.get_bid_range(state)
    
    def get_rewards(self, state):
        if not state.is_terminal():
            return 0
        else:
            quantity, face = self.game.unpack_action(state[0])
            if self.game.num_matches(hand_1, face) + self.game.num_matches(hand_2, face) >= quantity:
                if state[1]:
                    return 1
                else
                    return -1
            else
                if state[1]:
                    return -1
                else:
                    return 1
            

    def take_action(state, action):
        return self.game.act(state, action)
    
    def pbs2gamestate(self, pbs):
        return 

    def gamestate2public_state(self, game_state):
        return (game_state[0], game_state[1])