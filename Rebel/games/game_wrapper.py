#Wrappers for third party game implementations

class game_wrapper():
	'''
	Base class for game wrappers.

    Each node refers to a node of the game tree, which is represented as a tuple:
        ('root', actionID, actionID, ...)
    The zeroth element of nodes are always the string 'root'. After that, each
    successive entry is an action ID (some integer mapping to all possible
    actions of your game) corresponding to the actions that were played, in order.

    For example, in tic-tac-toe, the first actionID corresponds to player 1's first action, the second actionID
    was player 2's first action, etc.
	'''
    def __init__(self):
        game.num_actions = None #FILL IN: size of action space
        game.num_hands = None #FILL IN: number of infostates

	def get_legal_moves(self, node):
		'''
        Take a node and returns a list of all actionIDs possible from that node.
        '''
		raise NotImplementedError

    def node_to_state(self, node):
        '''
        Takes a node and returns the following tuple:
            (actionID of the last action taken, player who is about to play next)
        '''
        raise NotImplementedError

	def get_rewards(self, terminal_node):
		'''
        Given a terminal node, return a tuple of rewards for each player.

        In 2-player zero-sum games, this would be
            (player1's winnings, -1*player1's winnings)
        '''
		raise NotImplementedError

	def is_terminal(self,node):
		'''
        Given a node, return True if the node is terminal, that is, if there
        are no legal moves left (reached the end of the game). Return False otherwise.
        '''
		raise NotImplementedError

	def get_initial_beliefs(self, node):
		'''
        Return intital probability distribution over infostates.

        Should be a numpy array of size (1, self.num_hands)
        '''
		raise NotImplementedError

	def sample_hands(self, node):
		'''
        Return a randomly sampled hand. E.g. draw a set of cards.

        In perfect information games, there is only one possible 'hand', so you
        may just return 0.
        '''
		raise NotImplementedError
