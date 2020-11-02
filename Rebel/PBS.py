'''
Holder for node state values

'''

class PBS():
	'''
	Containts all public knowledge of knowledge and probability distribution for each players infostates
	For liars dice public is players turn, last bid, and probability distribution of each players hands
	'''
	def __init__(self, public_state, infostate_probs, policy = None):
		#representation of public state for the game
		self.public = public_state
		#list of probability matrices for each players infostate, (# players, infostate size)
		self.infostate_probs = infostate_probs 
		
		
	def update_infostate_probs(self, policy, action, player_number):
		'''
		policy is for the state - size (# infostates, actions)
		beyes update infostate posterior(state) ~ prior(state)*p(a|state)
		action is by the other player
		player_number is whose beliefs you're updating

		#TODO: these can all be done all at once by matrix multiplication
		'''
		new_infostate_probs = self.infostate_probs.copy()
		new_infostate_probs[player_number] = self.infostate_probs[player_number] * policy[action]/sum(self.infostate_probs[player_number] * policy[action])
		return(new_infostate_probs)


	def transition(self, action, policy, player_number):
		'''
		next PBS
		'''
		next_public_state = self.update_public_state(action, game) #TODO: 
		next_infostate_probs = self.update_infostate_probs(policy, action, player_number)
		return(PBS(next_public_state, next_infostate_probs))

	def vector(self):
		'''
		Numpy array representation -> input into value/policy net
		'''
		return(np.concat([self.public_state, *self.infostate_probs]))

#This is being moved into the game tree
def set_leaf_values(PBS, Policies, value_net):
	#If node ends in subgame but not the full game predict value with v_net
	
	if is_leaf(PBS):
		vs = value_net(PBS.vector)
		game_tree.add_node(pbs = PBS, value = vs)
	
	else:
		for action in possible_actions:
			set_leaf_values(PBS.transition(action, policy, 1-player_number))


