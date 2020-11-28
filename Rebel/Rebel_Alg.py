#psuedocode for the rebel overall rebel algorithm

import numpy as np
from game_tree import recursive_game_tree
from CFR import CFR

'''
Undefined functions in here
init_policy -> should probably be a just subgame method
update_policy -> CFR iteration, still needs to be implemented

Subgame (G) - Depth limited tree of game states
 - G.constuct_subgame -> (pseudocode)
 - G.set_leaf_values(policy, v_net) -> (pseudocode)
 - G.compute_ev(policy) -> not implemented
 - G.sample_leaf(policy) -> not implemented

 TODO: Move all policies into the subgame tree nodes

 '''


def ReBeL(PBS, game_wrapper, v_net, T=1000, p_net=None):
	"""
	The main ReBeL algorithm
	"""
	G = recursive_game_tree(PBS, game_wrapper)
	beliefs = np.ones(game_wrapper.num_hands) / game_wrapper.num_hands
	params = {'dcfr': False, 'linear_update': False}
	agent = CFR(game_wrapper, G, v_net, beliefs, params)
	D_v = [] #Training data for the value net
	while not PBS.is_terminal:
	#for i in range(2):
		#Build game tree going forward n actions from start node initializes random policy
		G.build_depth_limited_subgame(PBS)

		
		#Set values for the leaf nodes in the subgame using the value net if the node is not terminal
		#NOTE: policy is held in each game tree node
		G.set_leaf_values(v_net)

		#Initialize EV for the root PBS
		#Removed pi_t here cause policy now in the node
		#compute ev method defaults to root
		EV_PBS = G.compute_ev()

		#Sample a training iteration from linear distribution going from 0 to T
		t_sample = int(np.random.triangular(left = 0, mode = T, right = T))

		#Policy optimization loop using CFR etc 
		for t in range(t_warm+1, T):
			
			#Update policy using one iteration of CFR etc
			#pi_t = update_policy(G, pi_t)
			agent.step()
			pi_t = agent.last_strategies

			#update average policy
			#pi_bar = (t-1)/float(t+1) * pi_bar + 2/float(t+1) * pi_t
			pi_bar = agent.average_strategies

			#Recalculate leaf values using new policy
			#NOTE: removed pi_t here as arguement for same reason
			#NOTE: agent.step needs to modify the policy in place or we need a function to translate it
			G.set_leaf_values(v_net)

			#Update EV of root PBS
			#Changed this to work with compute EV as well
			EV_PBS = (t-1)/float(t+1) * EV_PBS + 2/float(t+1) * G.compute_ev()

			#Sample a leaf node for the next iteration
			#NOTE: Removed pi_t here. Policy should be stored in game tree
			if t == t_sample:
				next_PBS = G.sample_leaf() 

		#Add trainging data for value and policy net
		D_v.append(tuple(PBS.vector(), EV_PBS))

		#Note: No policy net
		#for node in G:
		#	D_p.append(tuple(node, pi_bar.get_policy(node)))

		PBS = next_PBS

	return(D_v)


