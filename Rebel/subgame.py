'''
class to build subgame for rebel
should work as a wrapper to interface rebel with whatecer game libraries we are using

Note:
- Leaf nodes are terminal in the subgame but not in the full game
- Policies map infostates to action probabilities
- input to the value net is a PBS


Methods to implent
 - G.set_leaf_values(policy, p_net)
 - G.compute_ev(policy)
 - G.sample_leaf()
'''

class PBS():
	'''
	Containts all public knowledge of knowledge and probability distribution for each players infostates
	For liars dice public is players turn, last bid, and probability distribution of each players hands
	'''
	def __init__(self, public_state, infostate_probs):
		#representation of public state for the game
		self.public = public_state
		#list of probability matrices for each players infostate, (len(#players)*infostate size)
		self.infostate_probs = infostate_probs 

	def update_public_state(self, action, game):
		'''
		given an action update the public state
		1. add action
		2. querry game for next deal etc.
		3. return new public state
		'''
		pass
		
	def update_infostate_probs(self, policy, action, player_number):
		'''
		policy is size (infostate, actions)
		beyes update infostate post(state) ~ prior(state)*p(a|state)
		action is by the other player
		player_number is whose beliefs you're updating
		'''
		new_infostate_probs = self.infostate_probs.copy()
		new_infostate_probs[player_number] = self.infostate_probs[player_number] * policy[action]/sum(self.infostate_probs[player_number] * policy[action])
		return(new_infostate_probs)


	def transition(self, action, policy, player_number):
		'''
		next PBS
		'''
		next_public_state = self.update_public_state(action, game)
		next_infostate_probs = self.update_infostate_probs(policy, action, player_number)
		return(PBS(next_public_state, next_infostate_probs))

	def vector(self):
		return(np.concat([self.public_state, *self.infostate_probs]))

#Note this is really rough. Still need to formalize what the game tree looks like
def set_leaf_values(PBS, Policies, value_net):
	#If node ends in subgame but not the full game predict value with v_net
	
	if is_leaf(PBS):
		#this needs to be set in some game tree graph
		vs = value_net(PBS.vector)
		game_tree.add_node(pbs = PBS, value = vs)
	
	else:
		#TODO: figure out how to get all possible actions
		for action in possible_actions:
			set_leaf_values(PBS.transition(action, policy, 1-player_number))




# class SubGame(Object):
# 	def __init__(self, PBS, Game):
# 		self.game_tree = _build_subgame_tree(PBS, Game)

# 	def _build_subgame_tree(PBS, Game):
# 		pass

# 	def set_leaf_values(self, infostate, policy, policy_net):
# 		for node in self.game_tree:
# 			if node.is_leaf:
# 				#Estimate the payoff for every non terminal leaf node
# 				v_si = policy_net()


	
# 	def compute_ev(self, policy):
# 		pass

# 	def sample_leaf(self, policy):
# 		pass