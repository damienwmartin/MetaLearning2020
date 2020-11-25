'''
Unit Tests for the Coin Game
'''
import unittest
import numpy as np
from game_wrappers import CoinGame


class TestCoinGame(unittest.TestCase):
    '''
    Requires any invalid inputs to result in throwing an exception (fail fast).
    '''
    def setUp():
        self.game = CoinGame()
        self.invalid_nodes = [('root', 0, 1),
                              (0, 1),
                              ('root', 1, 1, 0)]

    def tearDown():
        del self.game

    def test_get_legal_moves(self):
        with self.subTest('At root node'):
            node = ('root')
            expected_moves = [0,1]
            self.assertEqual(self.game.get_legal_moves(node), expected_moves)

        with self.subTest('Player 1 just sold'):
            node = ('root', 0)
            expected_moves = None
            self.assertEqual(self.game.get_legal_moves(node), expected_moves)

        with self.subTest('Player 1 just decided to bet'):
            node = ('root', 1)
            expected_moves = [0,1]
            self.assertEqual(self.game.get_legal_moves(node), expected_moves)

        with self.subTest('Player 2 just guessed heads'):
            node = ('root', 1, 0)
            expected_moves = None
            self.assertEqual(self.game.get_legal_moves(node), expected_moves)

        with self.subTest('Player 2 just guessed tails'):
            node = ('root', 1, 1)
            expected_moves = None
            self.assertEqual(self.game.get_legal_moves(node), expected_moves)

        for bad_node in self.invalid_nodes:
            with self.subTest('Invalid node_name', bad_node=bad_node):
                self.assertRaises(Exception, self.game.get_legal_moves, bad_node,
                              msg="An invalid node_name input did not result in a thrown exception when it should.")

    def test_take_action(self):
        with self.subTest('At root node, player 1 sells'):
            node = ('root')
            action = 0
            expected_result = ('root', 0)
            self.assertEqual(self.game.take_action(node), expected_result)

        with self.subTest('At root node, player 1 sells'):
            node = ('root')
            action = 1
            expected_result = ('root', 1)
            self.assertEqual(self.game.take_action(node), expected_result)

        with self.subTest('player 1 sold, player 2 attempts to guess heads'):
            node = ('root', 0)
            action = 0
            self.assertRaise(Exception, self.game.take_action, node, action,
                             msg="Actions are not possible from this state. Exception was not thrown when it should have been.")

        with self.subTest('player 1 sold, player 2 attempts to guess tails'):
            node = ('root', 0)
            action = 1
            self.assertRaise(Exception, self.game.take_action, node, action,
                             msg="Actions are not possible from this state. Exception was not thrown when it should have been.")

        with self.subTest('Player 1 chose to bet, player 2 guesses heads'):
            node = ('root', 1)
            action = 0
            expected_result = ('root', 1, 0)
            self.assertEqual(self.game.take_action(node), expected_result)

        with self.subTest('Player 1 chose to bet, player 2 guesses tails'):
            node = ('root', 1)
            action = 1
            expected_result = ('root', 1, 1)
            self.assertEqual(self.game.take_action(node), expected_result)

        with self.subTest('player 2 guessed heads, player 1 attempts another action 0'):
            node = ('root', 1, 0)
            action = 0
            self.assertRaise(Exception, self.game.take_action, node, action,
                             msg="Actions are not possible from this state. Exception was not thrown when it should have been.")

        with self.subTest('player 2 guessed heads, player 1 attempts another action 1'):
            node = ('root', 1, 0)
            action = 1
            self.assertRaise(Exception, self.game.take_action, node, action,
                             msg="Actions are not possible from this state. Exception was not thrown when it should have been.")

        with self.subTest('Invalid action'):
            node = ('root', 0, 1)
            action = 2
            self.assertRaises(self.game.take_action, node, action
                              msg="An invalid action input did not result in a thrown exception when it should.")

        for bad_node in self.invalid_nodes:
            with self.subTest('Invalid node_name', bad_node=bad_node):
                self.assertRaises(Exception, self.game.take_action, bad_node, 0
                              msg="An invalid node_name input did not result in a thrown exception when it should.")

    def test_node_to_number(self):
        with self.subTest('Root node'):
            node = ('root')
            expected_result = 0
            self.assertEqual(node, expected_result)

        with self.subTest('Player 1 sold'):
            node = ('root', 0)
            expected_result = 1
            self.assertEqual(node, expected_result)

        with self.subTest('Player 1 decided to bet'):
            node = ('root', 1)
            expected_result = 2
            self.assertEqual(node, expected_result)

        with self.subTest('Player 2 guessed heads'):
            node = ('root', 1, 0)
            expected_result = 3
            self.assertEqual(node, expected_result)

        with self.subTest('Player 1 decided to bet'):
            node = ('root', 1, 1)
            expected_result = 4
            self.assertEqual(node, expected_result)

        for bad_node in self.invalid_nodes:
            with self.subTest('Invalid node_name', bad_node=bad_node):
                self.assertRaises(Exception, self.game.node_to_number, bad_node,
                              msg="An invalid node_name input did not result in a thrown exception when it should.")


    def test_get_rewards(self):
        with self.subTest('P1 sells heads, plays tails; P2 always guesses heads')
            p1_strat = np.array([[1,0],
                                 [0,1]])

            p2_strat = np.array([[1,1],
                                 [0,0]])
            self.assertEqual(self.game.get_rewards(p1_strat, p2_strat), (0.75, -0.75))

        with self.subTest('P1 sells heads, plays tails; P2 always guesses tails')
            p1_strat = np.array([[1,0],
                                 [0,1]])

            p2_strat = np.array([[0,0],
                                 [1,1]])
            self.assertEqual(self.game.get_rewards(p1_strat, p2_strat), (0.25, -0.25))

        with self.subTest('P1 sells heads, plays tails; P2 guesses heads p=1/4, tails p=3/4)
            p1_strat = np.array([[1,0],
                                 [0,1]])

            p2_strat = np.array([[0.25,0.25],
                                 [0.75,0.75]])
            self.assertEqual(self.game.get_rewards(p1_strat, p2_strat), (0, 0))

        with self.subTest('P1 always sells; P2 guesses heads p=1/4, tails p=3/4)
            p1_strat = np.array([[0,0],
                                 [1,1]])

            p2_strat = np.array([[0.25,0.25],
                                 [0.75,0.75]])
            self.assertEqual(self.game.get_rewards(p1_strat, p2_strat), (0, 0))

    def test_node_to_state(self):
        with self.subTest('Root node'):
            node = ('root')
            expected_result = ('root', 0)
            self.assertEqual(self.game.node_to_state(node), expected_result)

        with self.subTest('Player 1 just sold'):
            #TODO: Confirm expected behavior is to say it's player 2's turn
            #      instead of throwing an exception
            node = ('root', 0)
            expected_result = (0, 1)
            self.assertEqual(self.game.node_to_state(node), expected_result)

        with self.subTest('Player 1 just decided to bet'):
            node = ('root', 1)
            expected_result = (1, 1)
            self.assertEqual(self.game.node_to_state(node), expected_result)

        with self.subTest('Player 2 just guessed heads'):
            #TODO: Confirm expected behavior is to say it's player 1's turn
            #      instead of throwing an exception
            node = ('root', 1, 0)
            expected_result = (0, 0)
            self.assertEqual(self.game.node_to_state(node), expected_result)

        with self.subTest('Player 2 just guessed tails'):
            #TODO: Confirm expected behavior is to say it's player 1's turn
            #      instead of throwing an exception
            node = ('root', 1, 1)
            expected_result = (1, 0)
            self.assertEqual(self.game.node_to_state(node), expected_result)

        for bad_node in self.invalid_nodes:
            with self.subTest('Invalid node_name', bad_node=bad_node):
                self.assertRaises(Exception, self.game.node_to_number, bad_node,
                              msg="An invalid node_name input did not result in a thrown exception when it should.")

    def test_is_terminal(self):
        with self.subTest('Root node'):
            node = ('root')
            self.assertFalse(self.game.is_terminal(node))

        with self.subTest('player 1 just sold'):
            node = ('root', 0)
            self.assertTrue(self.game.is_terminal(node))

        with self.subTest('Player 1 just decided to bet'):
            node = ('root', 1)
            self.assertFalse(self.game.is_terminal(node))

        with self.subTest('Player 2 just guessed heads'):
            node = ('root', 1, 0)
            self.assertTrue(self.game.is_terminal(node))

        with self.subTest('Player 2 just guessed tails'):
            node = ('root', 1, 1)
            self.assertTrue(self.game.is_terminal(node))

        for bad_node in self.invalid_nodes:
            with self.subTest('Invalid node_name', bad_node=bad_node):
                self.assertRaises(Exception, self.game.node_to_number, bad_node,
                              msg="An invalid node_name input did not result in a thrown exception when it should.")

    @unittest.SkipTest('Not yet Implemented')
    def test_sample_history(self):
        raise NotImplementedError

if __name__ == "__main__":
    unittest.main()
