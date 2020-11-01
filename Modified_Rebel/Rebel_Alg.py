#Just putting in psuedocode for the rebel overall rebel algorithm

import numpy

'''
Undefined functions in here
constuct_subgame
init_policy
compute_ev
update_policy
sample_leaf (probably want to move this into subgame methods)

Undefined classes
Subgame (G) - Depth limited tree of game states
 - G.set_leaf_values(policy, v_net)
 - G.compute_ev(policy)
 - G.sample_leaf(policy)
 

PBS - Probability distribution over true game states based on shared public knowledge

'''

def ReBeL(PBS, v_net, p_net, D_v, D_p, T):
	while not PBS.is_terminal:
		#Build game tree going forward n actions from start node
		G = construct_subgame(PBS)

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
			pi_t = update_policy(G, pi_t)

			#update average policy
			pi_bar = (t-1)/float(t+1) * pi_bar + 2/float(t+1) * pi_t

			#Recalculate leaf values using new policy
			G.set_leaf_values(pi_t, v_net)

			#Update EV of root PBS
			EV_PBS = (t-1)/float(t+1) * EV_PBS + 2/float(t+1) * compute_ev(G, pi_t)

			#Sample a leaf node for the next iteration
			if t == t_sample:
				next_PBS = sample_leaf(G, pi_t) 

		#Add trainging data for value and policy net
		D_v.append(tuple(PBS, EV_PBS))

		for node in G:
			D_p.append(tuple(node, pi_bar.get_policy(node)))

		PBS = next_PBS



