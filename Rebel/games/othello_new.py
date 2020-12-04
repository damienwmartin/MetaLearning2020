'''
Implementation of Othello.

Written from scratch. Needs additional testing.
'''
import numpy as np


class Othello():
    '''
    Single game of othello.

    Game representation:

        2 players, Black (id=1) and White (id=2). Black plays first.

        Game board is represented by a square 2D np.array corresponding
            to the rows and columns of a physical othello board.

        In outputs, a black piece is represented by 'O' (letter 'o') and
            a white piece is represented by 'X'.

        Because this is a perfect information game, the public belief
        state is the same as player's private beliefs, and hold total
        information about the game state.

        game_states are a np.array of size
                        (1, board_size*board_size+1)
            The first board_size*board_size entries corresponds to an
                unraveled board output.
                        blank space is represented with a 0
                        black piece is represented with a 1
                        white piece is represented with a 2
            The extra +1 entry denotes which player gets to play the next piece
                        (int) 0 for Black, 1 for White

        Action space is represented by np.array of size
                        (1, board_size*board_size+1)
            the first board_size*board_size entries correspend to placing
                piece at that cell. The extra +1 entry is for 'pass', when
                no valid moves are available this turn

        Reward for each player is how many pieces of their own color
            they have placed on the board.
    '''

    def __init__(self, board_size=4):
        '''
        Initialize a new game of othello.

        Game starts with
                X O
                O X
            in the middle of the board.

        board_size (int): game board size is board_size rows by
                            board_size columns.
                          board_size must be an even number >= 4.
        '''
        #check inputs
        assert board_size%2==0, 'board_size must be an even integer.'
        assert board_size>=4, 'board_size must be >= 4.'

        board = np.zeros((board_size, board_size))

        #top left start piece located at board[idx, idx]
        idx = board_size//2 - 2 #half -1 space, -1 0-indexing
        board[idx, idx] = 2 #top left X, white
        board[idx, idx] = 1 #top right O, black

        self.board_size = board_size

    def get_current_player_id(self, game_state=None):
        '''
        Return current player's id. 0 for black, 1 for white)

        game_state: game state to get player for. If None, then
                    the current game state stored in self is used.
        '''
        if not game_state:
            game_state = self.game_state

        player = game_state[0, -1]

        if player != 0 or player != 1:
            raise Exception('invalid player found')

        return player

    def get_current_player_name(self, game_state=None):
        '''
        Return current player's name as a string. 'Black' or 'White'.

        game_state: game state to get player for. If None, then
                    the current game state stored in self is used.
        '''
        if not game_state:
            game_state = self.game_state

        player = game_state[0, -1]

        if player == 0:
            return 'Black'
        elif player == 1:
            return 'White'
        else:
            raise Exception('Invalid player found')

    def get_board_size(self, game_state=None):
        '''
        Returns board_size of game represented by game_state.

        game_state: game state to get board size for. If None,
                    game state stored in self is used.
        '''
        if not game_state:
            game_state = self.game_state

        return int((game_state.shape[1]-1)**0.5)

    def actionID_to_cellCoords(self, action_id, game_state=None):
        '''
        Returns the (row, col) cordinates of the cell corresponding
        to action ID action_id.
        '''
        if not game_state:
            board_size = self.board_size
        else:
            board_size = self.get_board_size(game_state)
        row = action_id // board_size
        col = action_id % board_size

        return (row, col)

    def cellCoords_to_actionID(self, cell, game_state=None):
        '''
        Returns the (row, col) cordinates of the cell corresponding
        to cell

        cell: (row, col) ints

        game_state: game state containing the board to use. If None,
                    game state stored in self is used
        '''
        if not game_state:
            board_size = self.board_size
        else:
            board_size = self.get_board_size(game_state)
        row, col = cell
        return board_size*row + col


    def display_board(self, game_state=None):
        '''
        Pretty print a representation of the current game board,
        along with a string describing which player goes next.

        game_state: game state to display. If None, then the current
                    game state stored in self is displayed.
         '''
        if not game_state:
            game_state = self.game_state
        board = game_state[0,:-1]
        player = game_state[0,-1]



### Testing functions
'''
+---+ - + - +
| X | O |   |
+-+-+-+
|X|O| |
+-+-+-+
'''
