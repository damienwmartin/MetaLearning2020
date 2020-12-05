'''
Implementation of Othello.

Adapted from https://github.com/suragnair/alpha-zero-general/blob/master/othello

    Game representation:

        2 players, Black (id=[1) and White (id=1). Black plays first.

        Game board is represented by a square 2D np.array corresponding
            to the rows and columns of a physical othello board.

        In outputs, a black piece is represented by 'X' and
            a white piece is represented by 'O'.

        Because this is a perfect information game, the public belief
        state is the same as player's private beliefs, and hold total
        information about the game state.

        game_states are a np.array of size
                        (1, board_size*board_size+1)
            The first board_size*board_size entries corresponds to an
                unraveled board output.
                        blank space is represented with a 0
                        black piece is represented with a -1
                        white piece is represented with a 1
            The extra 1 entry denotes which player gets to play the next piece
                        (int) -1 for Black, 1 for White

        Action space is represented by np.array of size
                        (1, board_size*board_size+1)
            the first board_size*board_size entries correspend to placing
                piece at that cell. The extra entry is for 'pass', when
                no valid moves are available this turn

        Reward for each player is how many pieces of their own color
            they have placed on the board.
'''
import numpy as np


class OthelloBoard():
    '''
    Represents an othello board.

    board_size (int): game board size is board_size rows by
                        board_size columns.
                      board_size must be an even number >= 4.

    boards are indexed [row][col], unlike the original AlphaZero implementation.
    '''

    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, board_size=4):
        #check inputs
        assert board_size%2==0, 'board_size must be an even integer.'
        assert board_size>=4, 'board_size must be >= 4.'

        self.n = board_size

        # Create the empty board array.
        self.pieces = [[0]*board_size for _ in range(board_size)]

        # Set up the initial 4 pieces.
        self.pieces[int(self.n/2)-1][int(self.n/2)] = -1
        self.pieces[int(self.n/2)][int(self.n/2)-1] = -1
        self.pieces[int(self.n/2)-1][int(self.n/2)-1] = 1;
        self.pieces[int(self.n/2)][int(self.n/2)] = 1;

    def countDiff(self, color):
        """
        Counts the difference between # pieces of the given color

        color (int): 1 for white, -1 for black
        """
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
                if self[x][y]==-color:
                    count -= 1
        return count

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        (x,y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty source squares.
        if color==0:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        flips = [flip for direction in self.__directions
                      for flip in self._get_flips(move, direction, color)]
        assert len(list(flips))>0
        for x, y in flips:
            #print(self[x][y],color)
            self[x][y] = color

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in OthelloBoard._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        flips = [origin]

        for x, y in OthelloBoard._increment_move(origin, direction, self.n):
            #print(x,y)
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)):
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])

class OthelloGame():

    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = OthelloBoard(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = OthelloBoard(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = OthelloBoard(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = OthelloBoard(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = OthelloBoard(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")




class Othello():
    '''
    wrapper for othello functions to interface with our rebel implementations.

    Each node refers to a node of the game tree, represented as a tuple of
        ('root', action1, action2, action3, ..., 1 or -1)
        last 1 or -1 corresponds to which player goes next.

    Each action is an integer corresponding to the unraveled index of the square
        a piece was placed at.

        thus, action space is size (board_size*board_size + 1), with action
            board_size*board_size+1 corresponding to 'pass'
    '''

    def __init__(self, board_size=4):
        self.game = OthelloGame(n=board_size)
        self.board_size = board_size

        #required by rebel implementations
        self.num_actions = board_size*board_size+1
        game.num_hands = 1 #perfect information, so only one infostate

    def move_to_actionID(self, move):
        '''
        Returns actionID corresponding to placing a piece at a given row, col

        move: (row, col) of square to get actionID of.
        '''
        row, col = move
        return self.board_size*row + col

    def actionID_to_move(self, action):
        '''
        Returns the row, col corresponding to a given actionID

        action (int): actionID to convert
        '''
        return (int(action/self.board_size), action%self.board_size)

    def node_to_board(self, node):
        '''
        Gets the board corresponding to the histories of actions in node.
        '''
        player = 1
        board = OthelloBoard(self.n)
        if node == ('root'):
            return board
        #add pieces for each action after root
        for action in node[1:-1]:
            move = actionID_to_move(action)
            move = actionID_to_move(action)
            board.execute_move(move, player)
            player = -player
        return board

    def get_legal_moves(self, node):
        '''
        get list of valid next actionIDs, given the current node
        '''
        player = node[-1]
        board = self.node_to_board(node)
        valid_moves = board.get_legal_moves(player)
        legal_actions = [self.move_to_actionID(move) for move in valid_moves]
        return legal_actions

    def get_current_player(self, node):
        '''
        Return player who plays next, given a node
        '''
        if len(node)%2==0:
            return 1 #white goes on even turns
        else:
            return -1 #black goes first at root

    def node_to_state(self, node):
        '''
        Returns (last action ID, current player)

        Required by rebel implementation
        '''
        player = self.get_current_player(node)
        last_action = node[-1]
        return (last_action, player)


    def get_rewards(self, terminal_node):
        '''
        Returns rewards for each player (black, white) given a terminal node

        Currently reward is difference between number of pieces
        '''
        board = self.node_to_board(terminal_node)
        white_reward = board.countDiff(1)
        return (-white_reward, white_reward)

    def is_terminal(self, node):
        '''
        Returns True if node is termial. False otherwise
        '''
        player = self.get_current_player(node)
        board = self.node_to_board(node)
        return self.game.getGameEnded(board, player)

    def get_initial_beliefs(self):
        '''
        Required by rebel implementation.
        Because othello is perfect information game, there's only one possible 'hand'
        '''
        return np.array([1])

    def sample_hands(self):
        '''
        Required xy rebel implementation.
        Because othello is perfect information game, there's only one possible 'hand'
        '''
        return 0

    @staticmethod
    def display_board(self):
        raise NotImplementedError
        #TODO: make board output prettier






### Testing functions
'''
+---+ - + - +
| X | O |   |
+-+-+-+
|X|O| |
+-+-+-+
'''
