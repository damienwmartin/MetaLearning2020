import networkx as nx
import numpy as np
'''
Implentation for the game trees and wrappers for different game libraries
ReBeL only interacts with the subgame tree so if we place the wrapper here the rest should be application independant


Note:
- Leaf nodes are terminal in the subgame but not in the full game
'''

class recursive_game_tree():
	'''
	game tree implementation as NetworkX Directed Graph

	Node Ids are tuple with all actions taken from root e.g. ('root', 'raise', 'call', 'fold')
	
	Node Attributes
	-Depth
	-PBS
	-Policy

	'''
	def __init__ (self, PBS): 
		self.tree.add_node(('root',), depth=0, PBS=PBS)


	def build_depth_limited_subgame(self, depth_limit = 5, node_id = ('root',)):
		'''
		Recursively expand all nodes in the tree until you hit the depth limit
		Mark nodes which are subgame terminal
		'''
		node = self.tree.nodes[node_id]
		if node['depth'] < depth_limit:
			node['subgame_terminal'] = False
			self.expand_node(node_id)
			children = list(self.tree.successors(node_id))
			for child in children:
				self.build_depth_limited_subgame(depth_limit, child)
		if node['depth'] == depth_limit:
			node['subgame_terminal'] = True

	def initialize_random_policy(self):
		''''
		Run through the tree and set all node policies to be uniform over available actions
		Policy is (actions x infostates)
		'''
		for node_id, attributes in G.nodes.data():
			actions = self.game.get_legal_moves(attributes['PBS'])
			policy = {action:np.ones(self.game.num_infostates)/len(actions)}

	def set_leaf_values(self, policy, value_net, node_id=('root',)):
		#TODO: This is incomplete
		node = self.tree.nodes[node_id] 
		

		#Estimate values for nodes terminal in the subgame
		if node['subgame_terminal']:
			node['value'] = value_net(node['PBS'].vector())

		#Otherwise recursively rebuild the tree
		
		elif (i := node['depth']) < self.max_depth:
			for j, action in enumerate(legal_moves):
				self.tree.add_node((i+1, j),  {'depth':i+1, state = node['PBS'].transition(action)})
				self.tree.add_edge(node, (i+1, j), action = action)
				self.set_leaf_values(policy, value_net, self.tree.nodes[(i,j)]) #TODO the policy being passed here is wrong


	def expand_node(self, node_id):
		# Spawn a child node for every available actions
		legal_actions = self.game.get_legal_moves(pbs)
		
		for action in legal_actions:
			self.add_child(node_id, action)


	def add_child(self, node_id, action):
		'''
		get next PBS after taking action
		add PBS node to tree
		add edge from parent to chile
		'''
		node = self.tree.nodes[node_id]

		new_pbs = self.transition(node['PBS'], action)
		
		new_node_id = (*node_id, action)

		new_depth =  node['depth'] + 1

		self.tree.add_node(new_node_id, depth = new_depth, PBS = new_pbs)
		self.tree.add_edges(node_id, new_node_id, action=action)


	def transition(self, pbs, action):
		'''
		given a PBS and an actions spawn the next PBS
		'''
		#reach out to the game wrapper for the next public state
		next_public_state = self.game.take_action(action, pbs) 
		
		#Get the next distribution of infostates
		next_infostate_probs = pbs.update_infostate_probs(policy, action, player_number)

		#return a new updated pbs after taking action
		return(PBS(next_public_state, next_infostate_probs))




class PBS():
	'''
	Containts all public knowledge of knowledge and probability distribution for each players infostates
	For liars dice public is players turn, last bid, and probability distribution of each players hands
	'''
	def __init__(self, public_state, infostate_probs):
		
		#wrapper for the specific game we are playing
		self.game = game_wrapper
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

		'''
		new_infostate_probs = self.infostate_probs.copy()
		new_infostate_probs[player_number] = self.infostate_probs[player_number] * policy[action]/sum(self.infostate_probs[player_number] * policy[action])
		return(new_infostate_probs)


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
