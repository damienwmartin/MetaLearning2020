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
        """
        Returns a tuple (start, end) which represent all possible legal actions that can be taken from that state
        """
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
    
    def node_to_number(self, node):
        """
        Enumerates each of the nodes with a unique node_id. This node_id correspond to enumerating across each depth level (with the root node having an id of 0)
        """
        node_id = 0
        for i in range(node['depth']):
			node_id += binom(self.num_actions, i)
		last_action = node['id'][-2] if len(node['id']) > 2 else 0
		for i in range(node['depth'], last_action):
			node_id += (self.num_actions - i)
		node_id += last_action

        return node_id