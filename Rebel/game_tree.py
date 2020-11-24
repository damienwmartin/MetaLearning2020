import networkx as nx
import numpy as np
from scipy.special import binom

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
	-Depth: how deep
	-PBS
	-Policy
	-Terminal (both actually terminal and term)

	'''
	def __init__ (self, PBS, game): 
		self.tree = nx.DiGraph()
		self.game = game
		self.tree.add_node(('root',), depth=0, PBS=PBS, terminal = False, subgame_terminal = False)

	def build_depth_limited_subgame(self, depth_limit = 5, node_id = ('root',)):
		'''
		Recursively expand all nodes in the tree until you hit the depth limit
		Mark nodes which are subgame terminal
		'''
		node = self.tree.nodes[node_id]
		#initialize uniform_policy
		node['policy'] = np.ones(self.game.num_actions)/self.game.num_actions
		node['subgame_terminal'] = False
		if node['depth'] == depth_limit:
			node['subgame_terminal'] = True
		
		elif (node['depth'] < depth_limit) and (node['terminal'] == False):
			node['subgame_terminal'] = False
			self.expand_node(node_id)
			children = list(self.tree.successors(node_id))
			for child in children:
				self.build_depth_limited_subgame(depth_limit, child)





#NOTE: Moved into build subgame
#	def initialize_random_policy(self):
		'''
		Run through the tree and set all node policies to be uniform over available actions
		Policy is (actions x infostates)
		'''
#		for node_id, attributes in G.nodes.data():
#			actions = self.game.get_legal_moves(attributes['PBS'])
#			policy = {action:np.ones(self.game.num_infostates)/len(actions)}

	def set_leaf_values(self, value_net, node_id=('root',)):
		#TODO: This is incomplete
		node = self.tree.nodes[node_id] 
		print(node_id)
		

		if node['terminal']:
			print('setting terminal')
			node['value'] = self.game.get_rewards(node['PBS'])

		#Estimate values for nodes terminal in the subgame
		elif node['subgame_terminal']:
			print('setting subgame_terminal')
			node['value'] = value_net(node['PBS'].vector())

		#Otherwise recursively rebuild the tree
		
		else:
			for action in self.game.get_legal_moves(node['PBS']):
				self.add_child(node_id, action)
				self.set_leaf_values(value_net, (*node_id,action))


	def compute_ev(self, node_id = ('root',)):
		node = self.tree.nodes[node_id]
		
		if node['subgame_terminal'] or node['terminal']:
			return(node['value'])

		else:
			node['value']=0
			for action in self.game.get_legal_moves(['PBS']):
				node['value'] = node['value'] + node['policy'][action]*self.compute_ev((*node_id,action))
			return(node['value'])

	def sample_leaf(self):
		node_id = ('root',)
		node = self.tree.nodes[('root',)]
		random_player = np.random.choice([0,1])
		#TODO: figure out exactly how histories are represented
		h = self.game.sample_history(node)
		
		while not node['terminal'] and not node['subgame_terminal']:
			if random_player and (np.random.random()<.25):
				node = self.sample_child(node, h, random=True)
			else:
				node = self.sample_child(node,h)
			random_player =  not random_player

		return(node)


	def sample_child(self, node, infostate=None, random=False):
		#TODO: Make the policy a matrix instead of dict
		if random:
			probs = None

		elif infostate!=None:
			probs = node[:,'policy']
		else:
			probs = node['policy'].sum(axis=1) 

		children = list(self.tree.successors(node_id))
		child_id = np.random.choice(children, probs)
		return(self.tree.nodes[child_id])




	def expand_node(self, node_id):
		# Spawn a child node for every available actions
		PBS = self.tree.nodes[node_id]['PBS']
		legal_actions = self.game.get_legal_moves(PBS)
		
		for action in legal_actions:
			self.add_child(node_id, action)


	def add_child(self, node_id, action):
		'''
		get next PBS after taking action
		add PBS node to tree
		add edge from parent to child
		if child is terminal get the value from the game
		'''
		node = self.tree.nodes[node_id]

		new_pbs = self.transition(node['PBS'], action, node['policy'])
		
		new_node_id = (*node_id, action)

		new_depth =  node['depth'] + 1

		#If terminal get the payouts from the game and weight by probability of history
		if terminal := self.game.is_terminal(new_pbs):
			value = np.multiply(self.game.get_rewards(new_pbs), new_pbs.infostate_matrix()).sum().sum()

		self.tree.add_node(new_node_id, depth = new_depth, PBS = new_pbs, terminal=terminal)
		self.tree.add_edge(node_id, new_node_id, action=action, weight=node['policy'][action])


	def transition(self, pbs, action, policy = None):
		'''
		given a PBS and an actions spawn the next PBS
		'''
		#reach out to the game wrapper for the next public state
		next_public_state = self.game.take_action(pbs, action) 
		
		#Get the next distribution of infostates if policy exists
		next_infostate_probs = pbs.update_infostate_probs(policy, action)

		#return a new updated pbs after taking action
		return(PBS(next_public_state, next_infostate_probs))
	
	def __len__():
		return len(self.tree.nodes)
	



class PBS():
	'''
	Containts all public knowledge of knowledge and probability distribution for each players infostates
	For liars dice public is players turn, last bid, and probability distribution of each players hands
	TODO:Pull player number out into its own variable
	'''
	def __init__(self, public_state, infostate_probs):
		#representation of public state for the game
		self.public = public_state
		#list of probability matrices for each players infostate, (# players, infostate size)
		self.infostate_probs = infostate_probs 		
		
	def update_infostate_probs(self, policy, action):
		'''
		policy is for the state - size (# infostates, actions)
		beyes update infostate posterior(state) ~ prior(state)*p(a|state)
		action is by the other player
		player_number is whose beliefs you're updating

		'''
		player_number = self.public[0]
		new_infostate_probs = self.infostate_probs.copy()
		new_infostate_probs[player_number] = self.infostate_probs[player_number] * policy[action]/sum(self.infostate_probs[player_number] * policy[action])
		return(new_infostate_probs)

	def infostate_matrix(self):
		'''
		Joint probability of each players possible hands give infostate. 
		'''
		return(np.outer(self.infostate_probs[0].T,self.infostate_probs[1]))


	def vector(self):
		'''
		Numpy array representation -> input into value/policy net
		'''
		return(np.concat([self.public_state, *self.infostate_probs]))