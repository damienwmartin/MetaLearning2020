#Wrappers for third party game implementations

class game_wrapper():
	'''
	Base class for game wrappers
	'''
	def get_legal_moves(PBS):
		#Needs to take a game state and return legal moves for the next player 
		pass
	def get_rewards(PBS):
		#Needs to take a terminal game state and return rewards for each player
		pass
	def pbs2gamestate(PBS):
		#Converts PBS into a readable game state
		pass
	def gamestate2public_state(game_state):
		#Converts gamestate to PBS public state
		pass
	def take_action(game_state, action):
		#returns next gamestate
		pass

