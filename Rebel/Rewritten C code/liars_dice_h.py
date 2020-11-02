class LiarsDice_C():

    K_INITIAL_ACTION = -1

    def __init__(self, num_dice, num_faces):
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.total_num_dice = 2 * num_dice
        self.num_actions = 1 + total_num_dice*num_faces
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
        return (K_INITIAL_ACTION, 0)
    
    def get_bid_range(self, state):
        if state[0] == K_INITIAL_ACTION:
            return (0, self.num_actions - 1)
        else:
            return (state[0] + 1, self.num_actions)
    
    def is_terminal(self, state):
        return (state[0] == self.liar_call)
    
    def act(self, state, action):
        bid_range = self.get_bid_range(state):
        if (action < bid_range[0] or action >= bid_range[1])
            raise Exception("Action invalid")
        else:
            return (action, 1 - state.player)
