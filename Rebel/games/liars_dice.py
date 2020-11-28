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
    
    def node_to_number(self, node_name):
        """
        Enumerates each of the nodes with a unique node_id. This node_id correspond to enumerating across each depth level (with the root node having an id of 0)
        """

        if node_name == ('root',):
            return 0
        elif len(node_name) == 2:
            return node_name[1]
        else:
            node_id = 0
            for i in range(len(node_name) - 1):
                node_id += binom(self.num_actions, i)
            last_action = node_name[-2]
            for i in range(len(node_name) - 1, last_action):
                node_id += (self.num_actions - i)
            node_id += last_action

        return int(node_id)
    
    #NOTE: Not constant, possibly mask the total range in game tree
    def get_legal_moves(self, node_name):
        start, end = self.get_bid_ranges(node_name)
        return [i for i in range(start, end)]

    def sample_history(self, node_name, solver, random_action_prob, sampling_beliefs):

        # Samples a history from the PBS

        tree = solver.get_tree()
        path = []

        cur_node_name = node_name
        node = tree.nodes[node_name]
        br_sampler = np.random.randint(2)
        strategy = solver.get_sampling_strategy()
        beliefs = sampling_beliefs

        while not node['terminal']:
            node_id = self.node_to_number(cur_node_name)
            eps = np.random.uniform()
            state = self.node_to_state(cur_node_name)
            action_begin, action_end = self.get_bid_ranges(state)
            if state[1] == br_sampler and eps < random_action_prob:
                action = np.random.randint(action_begin, action_end)
            else:
                cur_beliefs = sampling_beliefs[state[1]]
                hand = np.random.choice(cur_beliefs.size, 1, p=cur_beliefs)
                policy = strategy[node_id][hand][0]
                action = np.random.choice(policy.size, 1, p=policy)[0]
                assert action >= action_begin and action < action_end
            
            policy = strategy[node_id]
            sampling_beliefs[state[1]] *= np.reshape(policy[:, action], (self.num_hands, ))
            
            # normalize_beliefs_inplace(sampling_beliefs[state[1]])
            sampling_beliefs[state[1]] += EPSILON
            sampling_beliefs[state[1]] /= np.sum(sampling_beliefs[state[1]], axis=0, keepdims=True)
            path.append((node_id, action))
            cur_node_name = cur_node_name + (action, )
            node = tree.nodes[cur_node_name]

        for node_id, action in path:
            policy = solver.get_belief_propagation_strategy()[node_id]

            beliefs[state[1]] = np.reshape(policy[:, action], (self.num_hands,))
            beliefs[state[1]] += EPSILON
            beliefs[state[1]] /= np.sum(beliefs[state[1]], axis=0, keepdims=True)
            # sampling_beliefs[state[1]] = np.reshape(policy[:, action], (self.num_hands, ))
            # normalize_beliefs_inplace(self.beliefs[state[1]])
            # sampling_beliefs += EPSILON
            # sampling_beliefs[state[1]] /= np.sum(sampling_beliefs[state[1]], axis=0, keepdims=True)
    
        return cur_node_name, beliefs

    def get_init_PBS(self):
        public_state = -1
        player_turn = 0
        infostate_probs = np.ones((2, self.num_faces**self.num_dice)) / (self.num_faces**self.num_dice)
        return(PBS(public_state, infostate_probs))
