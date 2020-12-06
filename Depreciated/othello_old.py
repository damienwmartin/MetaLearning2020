'''
Wrapper for AlphaZero's Othello for use with our implementation for ReBeL

Still a work in progress. Needs testing with finalized ReBeL implementation.
'''
import numpy as np
from game_tree import game_wrapper
from AlphaZero.othello.OthelloGame import OthelloGame


class Othello(game_wrapper):
    '''
    game_state is (board.pieces, player) where
        board.pieces is a (board_size x board_size) numpy array
        player is either -1 or 1, the player who will move next

    Default board_size is currently set to 6

    The public belief state representation is just a np.array of size
    (1, board_size*board_size+1), using the representation
    [board_size*board_size unraveled board entries, 1 player whose turn it is]
    '''

    def __init__(self, board_size=6):
        super().__init__()
        #Creates an othello board of size (board_size x board_size)
        self.game = OthelloGame(board_size)
        self.board_size = board_size

    def get_legal_moves(game_state):
        '''
        Takes a game state and returns the legal moves for the next player
        '''
	board = game_state[0]
        player = game_state[1]
	return self.game.getValidMoves(board, player)

    def get_rewards(game_state):
        '''
	Takes a terminal game state and return rewards for each player

        Right now reward is just set to the score

        Returns a (1x3) np.array of [player -1 reward, player 1 rewards, player -1 reward]
              so that rewards can be indexed by player using any of the following conventions:
                  "player -1" is either player 0, player -1, or player 2
        '''
	board = game_state[0]
        player = game_state[1]
        rewards = np.zeros((1,3)) #create empty np to fill with scores
        rewards[0,player] = self.game.getScore(board, player)
        rewards[0,-player] = self.game.getScore(board, -player)
        rewards[0,0] = rewards[0,-1]
        return rewards

    def pbs2gamestate(PBS):
        '''
	Converts PBS into a readable game state
        '''
        board = PBS[0,0:-1].reshape((self.board_size, self.board_size))
        player = PBS[0,-1]

    def gamestate2public_state(game_state):
        '''
	Converts gamestate to PBS public state
        '''
	board = game_state[0]
        player = game_state[1]
        return np.concat(np.ravel(board), np.array([player], axis=0)

    def take_action(game_state, action):
        '''
	Returns next public state
	'''
        board = game_state[0]
        player = game_state[1]
        return self.game.getNextState(board, player, action)
