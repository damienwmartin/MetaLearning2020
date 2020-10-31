'''
class to build subgame for rebel
should work as a wrapper to interface rebel with whatecer game libraries we are using

Methods to implent
 - G.set_leaf_values(policy, p_net)
 - G.compute_ev(policy)
 - G.sample_leaf()
'''

class InfoState(Object):
	def __init__(self, game_state, value = None)
		self.game_state = game_state
		self.value = value

class PBS(dict):
	def __init__(self, game_states, probabilities):
		#check there is a probability for each infostate
		assert(len(game_states)==len(probabilities))
		#check probabilities sum to one
		assert(sum(probabilities)==1)

		super().__init__()

		for state, probability in zip(game_states, probabilities):
			self[state] = probability





class SubGame(Object):
	def __init__(self, PBS, Game):


	def set_leaf_values(self, PBS, policy, policy_net):
		pass
	
	def compute_ev(self, policy):
		pass

	def sample_leaf(self, policy):
		pass