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
		pbs = node['PBS']
		num_actions = self.game.num_actions
		num_infostates = pbs.num_infostates()
		#initialize uniform_policy
		node['policy'] = np.ones((num_infostates, num_actions))/num_actions
		node['subgame_terminal'] = False
		if node['depth'] == depth_limit:
			node['subgame_terminal'] = True
		
		elif (node['depth'] < depth_limit) and (node['terminal'] == False):
			node['subgame_terminal'] = False
			self.expand_node(node_id)
			children = list(self.tree.successors(node_id))
			for child in children:
				self.build_depth_limited_subgame(depth_limit, child)
	

	def build_full_coin_game(self):
		self.tree.add_node(('root', 0), depth=1, terminal=True, subgame_terminal=True)
		self.tree.add_node(('root', 1), depth=1, terminal=False, subgame_terminal=False)
		self.tree.add_node(('root', 1, 0), depth=2, terminal=True, subgame_terminal=True)
		self.tree.add_node(('root', 1, 1), depth=2, terminal=True, subgame_terminal=True)


	def set_leaf_values(self, value_net, node_id=('root',), verbose = True):
		#TODO: This is incomplete
		node = self.tree.nodes[node_id] 
		pbs = node['PBS']

		if node['terminal']:
			payout_matrix = self.game.get_rewards(pbs)
			reward = np.dot(payout_matrix.T, pbs.infostate_matrix())
			
			if verbose:
				print(f"setting value for {node_id}")
				print('infostate_matrix * payout_matrix  = x')
				print(pbs.infostate_matrix(), ' * ', payout_matrix, ' = ', reward)
			node['value'] = reward

		#Estimate values for nodes terminal in the subgame
		elif node['subgame_terminal']:
			node['value'] = value_net(node['PBS'].vector())

		#Otherwise recursively rebuild the tree
			
		else:
			for action in self.game.get_legal_moves(node['PBS']):
				self.add_child(node_id, action)
				self.set_leaf_values(value_net, (*node_id,action))


	def compute_ev(self, node_id = ('root',), verbose = True):
		node = self.tree.nodes[node_id]
		
		if node['subgame_terminal'] or node['terminal']:
			return(node['value'])

		else:
			node['value']=0
			i=0
			for action in self.game.get_legal_moves(node['PBS']):
				i+=1
				next_state_ev = self.compute_ev((*node_id,action))
				node['value'] = node['value'] + sum(node['policy'][:,action]) * next_state_ev
				if verbose:
					print(f"Adding value to {node_id} for {action}")
					print('p(next_state) * next_state(value)  = x')
					print(node['policy'][:,action], ' * ', next_state_ev, ' = ', sum(node['policy'][:,action]) * next_state_ev)
			#renormalize
			node['value'] = node['value'] / node['PBS'].num_infostates()
			return(node['value'])

	def update_policy(self):
		#cfr going in here
		pass


	def sample_leaf(self):
		node_id = ('root',)
		node = self.tree.nodes[('root',)]
		pbs = node['PBS']

		#Choose a player to act randomly
		random_player = np.random.choice([0,1])
		
		#sample a history (starting hand for each player)
		infostates = [np.random.randint(pbs.num_infostates(i)) for i in range(2)]
		
		while not node['terminal'] and not node['subgame_terminal']:
			#Get infostate for current_player
			infostate = infostates[node['PBS'].player_turn]
			#sample some actions randomly to encourage exploration
			if random_player and (np.random.random()<.25):
				node_id = self.sample_child(node_id, infostate, random=True)
			#otherwise sample a child according to the policy at that infostate
			else:
				node_id = self.sample_child(node_id, infostate)
			node = self.tree.nodes[node_id]
		#returns leaf PBS and the list of actions that led there
		return(node['PBS'])


	def sample_child(self, node_id=('root',), infostate=None, random=False):
		node = self.tree.nodes[node_id]
		pbs = node['PBS']

		if random:
			probs = None

		else:
			probs = node['policy'][infostate] 


		children = list(self.tree.successors(node_id))
		child_id = children[np.random.choice(len(children), p=probs)]
		return(child_id)




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
		terminal = self.game.is_terminal(new_pbs)
		#If terminal get the payouts from the game and weight by probability of history
		if terminal:
			value = np.multiply(self.game.get_rewards(new_pbs), new_pbs.infostate_matrix()).sum().sum()

		self.tree.add_node(new_node_id, depth = new_depth, PBS = new_pbs, terminal=terminal)
		self.tree.add_edge(node_id, new_node_id, action=action, weight=sum(node['policy'][:,action]))


	def transition(self, pbs, action, policy = None):
		'''
		given a PBS and an actions spawn the next PBS
		'''
		#reach out to the game wrapper for the next public state
		next_public_state = self.game.take_action(pbs, action) 
		
		#Get the next distribution of infostates if policy exists
		next_infostate_probs = pbs.update_infostate_probs(policy, action)

		#Get the next player
		player_number = 1 - pbs.player_turn

		#return a new updated pbs after taking action
		return(PBS(next_public_state, next_infostate_probs, player_number))
	
	def __len__(self):
		return len(self.tree.nodes)
	



class PBS():
	'''
	Containts all public knowledge of knowledge and probability distribution for each players infostates
	For liars dice public is players turn, last bid, and probability distribution of each players hands
	TODO:Pull player number out into its own variable
	'''
	def __init__(self, public_state, infostate_probs, player = 0):
		self.player_turn = player
		#representation of public state for the game
		self.public = public_state
		#list of probability matrices for each players infostate, (# players, infostate size)
		self.infostate_probs = infostate_probs 		
		
	def num_infostates(self, player=None):
		'''
		for just the current player
		'''
		if player == None:
			player = self.player_turn
		return(len(self.infostate_probs[player]))

	def update_infostate_probs(self, policy, action, verbose = True):
		'''
		policy is for the state - size (# infostates, actions)
		beyes update infostate posterior(state) ~ prior(state)*p(a|state)
		action is by the other player
		player_number is whose beliefs you're updating

		'''
		player_number = self.player_turn
		new_infostate_probs = self.infostate_probs.copy()

		if verbose:
			print('new_infostate_probs = infostate_probs[player_number] * policy[:,action]')

		posterior = self.infostate_probs[player_number] * policy[:,action]
		new_infostate_probs[player_number] = posterior/sum(posterior)
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
		return(np.concatenate([[self.player_turn], [self.public], *self.infostate_probs]))