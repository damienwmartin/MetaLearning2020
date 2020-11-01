import networkx as nx


class game_wrapper():
	
	def get_legal_moves(game_state):
		pass

	def get_rewards(game_state):
		pass




class game_tree():
	'''
	Game tree implemented as a networkx directed graph
	Node names are (depth from root, child number)
	Node attributes are dict -> depth, state
	'''
	def __init__(self, game_wrapper):
		self.game = game_wrapper
		self.tree = nx.DiGraph()
	
	#build subgame tree with depth t
	def construct_subgame(self, PBS, depth):
		self.tree.add_node((0, 0), depth=0, state=PBS)
		node_queue = [(0,0)]
		next_node_queue = []

		#for every range up to desired depth
		for i in range(depth+1):
			#Check every node in the queue
			for node in node_queue:

				#Mark as terminal if there are no legal moves
				legal_moves = self.game.get_legal_moves(game_state)

				if legal_moves == None:
					self.tree.nodes[node]['terminal'] = True

				#otherwise expand the tree until we hit the desired depth
				else:
					self.tree.nodes[node]['terminal'] = False
					if i != depth:
						for j, action in enumerate(legal_moves):
							self.tree.add_node((i+1, j),  {'depth':i+1})
							self.tree.add_edge(node, (i+1, j), action = action)
							next_node_queue.append((i+1, j))
			node_queue = next_node_queue
	
	def set_leaf_values(policy, value_net):
		raise (NotImplementedError)

	def sample_leaf(self, policy)
		raise(NotImplementedError)
	
	def compute_ev(policy):
		raise(NotImplementedError)




