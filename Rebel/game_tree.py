import networkx as nx
'''
Implentation for the game trees and wrappers for different game libraries
ReBeL only interacts with the subgame tree so if we place the wrapper here the rest should be application independant
'''

class game_tree():
	'''
	Game tree implemented as a networkx directed graph
	Node names are (depth from root, child number)
	Node attributes are dict -> depth, state (PBS)
	Edge atributes are actions 
	'''
	def __init__(self, game_wrapper):
		self.game = game_wrapper
		self.tree = nx.DiGraph()
	
	#build subgame tree with depth t
	def construct_subgame(self, PBS, depth):
		self.tree.add_node((0, 0), depth=0, PBS=PBS)
		node_queue = [(0,0)]

		#for every range up to desired depth
		for i in range(depth+1):
			next_node_queue = []
			#Check every node in the queue
			for xy in node_queue:
				node = self.tree.node(xy)
				#Mark as terminal if there are no legal moves
				game_state = self.game.pbs2gamestate(node['PBS'])
				legal_moves = self.game.get_legal_moves(game_state)

				if legal_moves == None:
					node['terminal'] = True
					node['subgame_terminal'] = False
					node['value'] = self.game.get_rewards(game)

				#otherwise expand the tree until we hit the desired depth
				else:
					node['terminal'] = False
					if i != depth:
						node['subgame_terminal']=False
						for j, action in enumerate(legal_moves):
							self.tree.add_node((i+1, j),  {'depth':i+1, state = node['PBS'].transition(action)})
							self.tree.add_edge(node, (i+1, j), action = action)
							next_node_queue.append((i+1, j))
					else:
						node['subgame_terminal'] = True
			node_queue = next_node_queue.copy()
	
	def set_leaf_values(self, policy, value_net, node=None):
		#start at root if no node is given
		if node == None	
			node = self.tree.nodes[(0,0)]

		#
		if node['subgame_terminal']:





	def sample_leaf(self, policy)
		raise(NotImplementedError)
	
	def compute_ev(policy):
		raise(NotImplementedError)


class recursive_game_tree():
	#Much better way to implement this
	def __init__ (self, PBS): 
		self.tree.add_node((0, 0), depth=0, PBS=PBS, )

	def expand_node(node = None):
		if node == None:
			pass

	def update_node(node = None):
		


