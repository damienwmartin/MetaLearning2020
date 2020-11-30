import numpy as np
class player():
	def __init__(self, game, hand):
		self.game = game
		self.hand = hand

	def take_action(self, previous_action):
		action = None
		return(action)

class random_player(player):
	def __init__(self, game, hand):
		super().__init__(game, hand)
		self.cur_node_id = ('root',)
	def take_action(self, previous_action):
		if previous_action!=None:
			self.cur_node_id = (*self.cur_node_id, previous_action)
		legal_actions = self.game.get_legal_moves(self.cur_node_id)
		if len(legal_actions) == 0:
			print('No Action: ',self.cur_node_id)
		action = np.random.choice(legal_actions)
		self.cur_node_id = (*self.cur_node_id, action)
		return(action)


class rebel_player(player):
    '''
    Rebel player - Takes a game, the hand it was dealt and a trained value network
    take_action - takes the last action by the other player and returns its response
    '''
    def __init__(self, game, hand, v_net, solver = None, T=1000, depth_limit = 5):
        if solver is None:
            initial_beliefs = game.get_initial_beliefs()
            params = {'dcfr': False, 'linear_update': False}
            solver = CFR(game, value_net, initial_beliefs, params)
        
        cur_node = solver.tree.nodes[('root', )]
        cur_node_id = ('root', )


        solver.build_depth_limited_subgame(cur_node_id, depth_limit=5)
        t_sample = np.random.randint(T)

        for t in range(t_sample):                
            solver.step(t % 2)            
        
        self.solver = solver
        self.cur_node_id = cur_node_id
        self.hand = hand
        self.depth_limit = depth_limit


    def take_action(self, previous_action = None):
        if previous_action != None:
            #previous actions is the other players last move, None if this is the first move of the game
            self.update_current_node(previous_action)

        #Grab the policy at the current node and use it to sample an action
        node = self.solver.tree.nodes[self.cur_node_id]
        policy = node['cur_strategy'][self.hand][0]
        action = np.random.choice(policy.size, 1, p=policy)[0]
        assert action in self.game.get_legal_moves(cur_node_name)
        
        self.update_current_node(action)
        return(action)
        

    def update_current_node(self, action):
        #update current node and resolve subgame if we hit the depth limit
        self.cur_node_id = (*self.cur_node_id, previous_action)
        if self.cur_node_id['subgame_terminal']:
            self.solver.build_depth_limited_subgame(cur_node_id, depth_limit = self.depth_limit)
            t_sample = np.random.randint(T)
            for t in range(t_sample):                
                self.solver.step(t % 2)            