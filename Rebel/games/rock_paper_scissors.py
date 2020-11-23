import numpy as np
class RPS():
	'''
	RPS
	Normal rock paper scissors rules apply
	Plus every round one of R,P,S is chosen
	If you win with that throw you get double points

	1 point for win
	0 points for tie
	2 points for a win with the special hand
	'''
	def __init__(self):
		self.num_actions = 3
		self.num_infostates = 9


	def get_legal_moves(self, PBS):
		'''
		0 -> Rock
		1 -> Paper
		2 -> Scissors
		'''
		return([0,1,2])

	def get_rewards(self, PBS):
		#Needs to take a terminal game state and return rewards for each player
		payout_matrix = np.array([[0, -1, 1],
								  [1, 0, -1],
								  [-1, 1, 0]])
		special_hand = PBS.public
		payout_matrix[:,special_hand] = payout_matrix[:,special_hand]*2
		payout_matrix[special_hand,:] = payout_matrix[special_hand,:]*2
		reward = np.multiply(payout_matrix, PBS.infostate_matrix).sum().sum()
		return(reward)

	
	def sample_history(self, PBS):
		#samples a history from the PBS
		hand1 = np.random.choice([0,1,2], p=PBS.infostate_probs[1])
		hand2 = np.random.choice([0,1,2], p=PBS.infostate_probs[0])
		return(np.array(hand1, hand2))

		

	def take_action(self, PBS, action):
		#returns next public state, only change is the players turn
		new_public_state = PBS.public.copy()
		new_public_state[0] = 1 - new_public_state[0]
		return(new_public_state)

	def is_terminal(self, PBS):
		return(PBS.public[0]==1)


		
