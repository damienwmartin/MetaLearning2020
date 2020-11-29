from game_wrappers import game_wrapper
from game_tree import PBS
import numpy as np
from scipy.special import binom

# import LiarsDice_C from Modified C Code?

EPSILON = 1e-100

class LiarsDice(game_wrapper):
    """
    Simple Wrapper for Liar's Dice that is an extension of our generic game_wrapper class
    PBS are given as (action_id, player_id) pairs
    game_states are given as (action_id, player_id, hand_1, hand_2)
    """
    
    
    #NOTE: I think this needs to work with PBS distribution
    def get_rewards(self, state):
        if not state.is_terminal():
            return 0
        else:
            quantity, face = self.game.unpack_action(state[0])
            if self.game.num_matches(hand_1, face) + self.game.num_matches(hand_2, face) >= quantity:
                if state[1]:
                    return 1
                else:
                    return -1
            else:
                if state[1]:
                    return -1
                else:
                    return 1

    def __init__(self, num_dice, num_faces):
        self.K_INITIAL_ACTION = -1
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.total_num_dice = 2 * num_dice
        self.num_actions = 1 + num_dice*num_faces
        self.num_hands = num_faces**num_dice
        self.liar_call = self.num_actions - 1 # Action ID for calling BS
        self.wild_face = num_faces -  1 # Face ID for the face considered "wild"
    
    def max_depth(self):
        """
        An upper bound for how deep the game can be - can only call distinct hands
        """
        return self.num_actions + 1
    
    def unpack_action(self, action):
        """
        Given an action id, returns the actual features of the action
        """
        quantity = action // self.num_faces + 1
        face = action % self.num_faces
        return quantity, face

    def num_matches(self, hand, face):
        """
        NOTE: Need to remember to ask what this is doing
        Hands are basically represented as numbers in base (self.num_faces)
        """
        matches = 0
        for i in range(self.num_dice):
            dice_face = hand % self.num_faces
            matches += (dice_face == face or dice_face == self.wild_face)
            hand = hand // self.num_faces
        return matches
    
    def get_initial_state(self):
        """
        States are represented as tuples of (action_id, player_id)
        """
        return (self.K_INITIAL_ACTION, 0)
    
    def get_bid_ranges(self, node_name):
        """
        Returns a tuple (start, end) which represent all possible legal actions that can be taken from that state
        """

        state = self.node_to_state(node_name)
        if state[0] == self.K_INITIAL_ACTION:
            return (0, self.num_actions - 1)
        else:
            return (state[0] + 1, self.num_actions)
    
    #Note changed this to run with PBS
    def is_terminal(self, node_name):
        """
        Determines whether or not the state inputted is a terminal state
        """
        return (node_name[-1] == self.liar_call)
    
    def act(self, state, action):
        bid_range = self.get_bid_range(state)
        if (action < bid_range[0] or action >= bid_range[1]):
            raise Exception("Action invalid")
        else:
            return (action, 1 - state.player)

    def take_action(self, node_name, action):
        '''
        Version of act taking in PBS
        '''
        bid_range = self.get_bid_ranges(node_name)
        if (action < bid_range[0] or action >= bid_range[1]):
            raise Exception("Action invalid")
        return node_name + (action, )

        
    def iter_at_node(self, node_id):
        last_action = node_id[-1]
        for i in range(self.num_actions):
            if i > last_action:
                yield i, node_id + i
            else:
                yield i, None
    
    def node_to_state(self, node_id):
        """
        Converts a node into a tuple of (last_action, current_player)
        """
        if len(node_id) == 1:
            return self.get_initial_state()
        else:
            return (node_id[-1], (len(node_id) - 1) % 2)
    
    #NOTE: Not constant, possibly mask the total range in game tree
    def get_legal_moves(self, node_name):
        start, end = self.get_bid_ranges(node_name)
        return [i for i in range(start, end)]
    
    def get_initial_beliefs(self):
        return np.ones((2, self.num_hands)) / self.num_hands

    def get_init_PBS(self):
        public_state = -1
        player_turn = 0
        infostate_probs = np.ones((2, self.num_faces**self.num_dice)) / (self.num_faces**self.num_dice)
        return(PBS(public_state, infostate_probs))
