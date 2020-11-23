#psuedocode for the rebel overall rebel algorithm

import numpy as np
from game_tree import game_tree
from .CFR import CFR

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


def ReBeL(PBS, v_net, p_net, D_v, D_p, T, game_wrapper):
	while not PBS.is_terminal:
		#Build game tree going forward n actions from start node
		G = game_tree(game_wrapper)
		beliefs = np.ones(game_wrapper.num_hands) / game_wrapper.num_hands
		agent = CFR(game_wrapper, game_tree, v_net, beliefs, params)
		G.construct_subgame(PBS)

		#initialize policy using the policy network
		pi_bar, pi_t = init_policy(G, p_net)

		#Set values for the leaf nodes in the subgame using the value net if the node is not terminal
		G.set_leaf_values(pi_t, v_net)

		#Initialize EV for the root PBS
		EV_PBS = compute_ev(G, pi_t)

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
			G.set_leaf_values(pi_t, v_net)

			#Update EV of root PBS
			EV_PBS = (t-1)/float(t+1) * EV_PBS + 2/float(t+1) * compute_ev(G, pi_t)

			#Sample a leaf node for the next iteration
			if t == t_sample:
				next_PBS = G.sample_leaf(pi_t) 

		#Add trainging data for value and policy net
		D_v.append(tuple(PBS, EV_PBS))

		for node in G:
			D_p.append(tuple(node, pi_bar.get_policy(node)))

		PBS = next_PBS



